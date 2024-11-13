#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

#[cfg(feature = "cuda")]
use cryptography_cuda::{
    device::memory::HostOrDeviceSlice, intt_batch, lde_batch, lde_batch_multi_gpu,
    transpose_rev_batch, types::*,
};
use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

#[cfg(all(feature = "cuda", any(test, doctest)))]
pub static GPU_INIT: once_cell::sync::Lazy<std::sync::Arc<std::sync::Mutex<u64>>> =
    once_cell::sync::Lazy::new(|| std::sync::Arc::new(std::sync::Mutex::new(0)));

#[cfg(all(feature = "cuda", any(test, doctest)))]
fn init_gpu() {
    use cryptography_cuda::init_cuda_rs;

    let mut init = GPU_INIT.lock().unwrap();
    if *init == 0 {
        println!("Init GPU!");
        init_cuda_rs();
        *init = 1;
    }
}

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    #[cfg(not(feature = "cuda"))]
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        Self::from_values_cpu(
            values,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values_cpu(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        // #[cfg(any(not(feature = "cuda"), not(feature = "batch")))]
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs_cpu(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = values[0].len();
        let log_n = log2_strict(degree);

        if log_n > 1 && log_n + rate_bits > 1 && values.len() > 0 {
            #[cfg(any(test, doctest))]
            init_gpu();

            let _num_gpus: usize = std::env::var("NUM_OF_GPUS")
                .expect("NUM_OF_GPUS should be set")
                .parse()
                .unwrap();

            Self::from_values_gpu(
                values.as_slice(),
                rate_bits,
                blinding,
                cap_height,
                timing,
                fft_root_table,
                log_n,
                degree,
            )
        } else {
            Self::from_values_cpu(
                values,
                rate_bits,
                blinding,
                cap_height,
                timing,
                fft_root_table,
            )
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_values_gpu(
        values: &[PolynomialValues<F>],
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        _fft_root_table: Option<&FftRootTable<F>>,
        log_n: usize,
        degree: usize,
    ) -> Self {
        let output_domain_size = log_n + rate_bits;

        let num_gpus: usize = std::env::var("NUM_OF_GPUS")
            .expect("NUM_OF_GPUS should be set")
            .parse()
            .unwrap();

        let total_num_of_fft = values.len();
        println!("total_num_of_fft: {:?}", total_num_of_fft);
        // println!("fft_size: {:?}", log_n);

        let total_num_input_elements = total_num_of_fft * (1 << log_n);
        let total_num_output_elements = total_num_of_fft * (1 << output_domain_size);

        let mut gpu_input: Vec<F> = values
            .into_iter()
            .flat_map(|v| v.values.iter().cloned())
            .collect();

        let mut device_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, total_num_input_elements).unwrap();

        let _ret = device_data.copy_from_host(&gpu_input);

        let mut cfg_ntt = NTTConfig::default();
        cfg_ntt.are_inputs_on_device = true;
        cfg_ntt.are_outputs_on_device = true;
        cfg_ntt.batches = total_num_of_fft as u32;

        intt_batch(0, device_data.as_mut_ptr(), log_n, cfg_ntt.clone());

        let mut cfg_lde = NTTConfig::default();
        cfg_lde.batches = total_num_of_fft as u32;
        cfg_lde.extension_rate_bits = rate_bits as u32;
        cfg_lde.are_inputs_on_device = true;
        cfg_lde.are_outputs_on_device = true;
        cfg_lde.with_coset = true;
        cfg_lde.is_multi_gpu = true;

        let mut device_output_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, total_num_output_elements).unwrap();

        if num_gpus == 1 {
            let _ = timed!(
                timing,
                "LDE on 1 GPU",
                lde_batch(
                    0,
                    device_output_data.as_mut_ptr(),
                    device_data.as_mut_ptr(),
                    log_n,
                    cfg_lde.clone()
                )
            );
        } else {
            let _ = timed!(
                timing,
                "LDE on multi GPU",
                lde_batch_multi_gpu::<F>(
                    device_output_data.as_mut_ptr(),
                    device_data.as_mut_ptr(),
                    num_gpus,
                    cfg_lde.clone(),
                    log_n,
                )
            );
        }

        let mut coeffs_1d = vec![F::ZERO; total_num_input_elements];
        device_data
            .copy_to_host(coeffs_1d.as_mut_slice(), total_num_input_elements)
            .unwrap();

        let chunk_size = 1 << log_n;
        let coeffs_batch: Vec<PolynomialCoeffs<F>> = coeffs_1d
            .chunks(chunk_size)
            .map(|chunk| PolynomialCoeffs {
                coeffs: chunk.to_vec(),
            })
            .collect();

        let mut cfg_trans = TransposeConfig::default();
        cfg_trans.batches = total_num_of_fft as u32;
        cfg_trans.are_inputs_on_device = true;
        cfg_trans.are_outputs_on_device = true;

        let mut device_transpose_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, total_num_output_elements).unwrap();

        let _ = timed!(
            timing,
            "transpose",
            transpose_rev_batch(
                0 as i32,
                device_transpose_data.as_mut_ptr(),
                device_output_data.as_mut_ptr(),
                output_domain_size,
                cfg_trans
            )
        );

        let mt = timed!(
            timing,
            "Merkle tree with GPU data",
            MerkleTree::new_from_gpu_leaves(
                &device_transpose_data,
                1 << output_domain_size,
                total_num_of_fft,
                cap_height
            )
        );

        drop(device_transpose_data);
        drop(device_output_data);
        drop(device_data);

        assert_eq!(coeffs_batch.len(), values.len());

        Self {
            polynomials: coeffs_batch,
            merkle_tree: mt,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs_cpu(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();

        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_from_2d(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        Self::from_coeffs_cpu(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {

        let pols = polynomials.len();
        let degree = polynomials[0].len();
        let log_n = log2_strict(degree);

        if log_n + rate_bits > 1 && polynomials.len() > 0 {
            #[cfg(any(test, doctest))]
            init_gpu();

            let _num_gpus: usize = std::env::var("NUM_OF_GPUS")
                .expect("NUM_OF_GPUS should be set")
                .parse()
                .unwrap();

            let merkle_tree = Self::from_coeffs_gpu(
                &polynomials,
                rate_bits,
                blinding,
                cap_height,
                timing,
                fft_root_table,
                log_n,
                degree,
            );

            return Self {
                polynomials,
                merkle_tree,
                degree_log: log2_strict(degree),
                rate_bits,
                blinding,
            };
        } else {
            Self::from_coeffs_cpu(
                polynomials,
                rate_bits,
                blinding,
                cap_height,
                timing,
                fft_root_table,
            )
        }
    }

    #[cfg(feature = "cuda")]
    fn from_coeffs_gpu(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        _fft_root_table: Option<&FftRootTable<F>>,
        log_n: usize,
        _degree: usize,
    ) -> MerkleTree<F, <C as GenericConfig<D>>::Hasher> {
        let salt_size = if blinding { SALT_SIZE } else { 0 };
        // println!("salt_size: {:?}", salt_size);
        let output_domain_size = log_n + rate_bits;

        let num_gpus: usize = std::env::var("NUM_OF_GPUS")
            .expect("NUM_OF_GPUS should be set")
            .parse()
            .unwrap();
        // let num_gpus: usize = 1;
        // println!("get num of gpus: {:?}", num_gpus);
        let total_num_of_fft = polynomials.len();
        // println!("total_num_of_fft: {:?}", total_num_of_fft);

        let num_of_cols = total_num_of_fft + salt_size; // if blinding, extend by salt_size
        let total_num_input_elements = total_num_of_fft * (1 << log_n);
        let total_num_output_elements = num_of_cols * (1 << output_domain_size);

        let mut gpu_input: Vec<F> = polynomials
            .into_iter()
            .flat_map(|v| v.coeffs.iter().cloned())
            .collect();

        let mut cfg_lde = NTTConfig::default();
        cfg_lde.batches = total_num_of_fft as u32;
        cfg_lde.extension_rate_bits = rate_bits as u32;
        cfg_lde.are_inputs_on_device = true;
        cfg_lde.are_outputs_on_device = true;
        cfg_lde.with_coset = true;
        cfg_lde.is_multi_gpu = true;
        cfg_lde.salt_size = salt_size as u32;

        let mut device_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, total_num_input_elements).unwrap();

        let _ret = device_data.copy_from_host(&gpu_input);

        let mut device_output_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, total_num_output_elements).unwrap();
        if num_gpus == 1 {
            let _ = timed!(
                timing,
                "LDE on 1 GPU",
                lde_batch(
                    0,
                    device_output_data.as_mut_ptr(),
                    device_data.as_mut_ptr(),
                    log_n,
                    cfg_lde.clone()
                )
            );
        } else {
            let _ = timed!(
                timing,
                "LDE on multi GPU",
                lde_batch_multi_gpu::<F>(
                    device_output_data.as_mut_ptr(),
                    device_data.as_mut_ptr(),
                    num_gpus,
                    cfg_lde.clone(),
                    log_n,
                )
            );
        }

        let mut cfg_trans = TransposeConfig::default();
        cfg_trans.batches = num_of_cols as u32;
        cfg_trans.are_inputs_on_device = true;
        cfg_trans.are_outputs_on_device = true;

        let mut device_transpose_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, total_num_output_elements).unwrap();

        let _ = timed!(
            timing,
            "transpose",
            transpose_rev_batch(
                0 as i32,
                device_transpose_data.as_mut_ptr(),
                device_output_data.as_mut_ptr(),
                output_domain_size,
                cfg_trans
            )
        );

        let mt = timed!(
            timing,
            "Merkle tree with GPU data",
            MerkleTree::new_from_gpu_leaves(
                &device_transpose_data,
                1 << output_domain_size,
                num_of_cols,
                cap_height
            )
        );

        drop(device_transpose_data);
        drop(device_output_data);

        mt
    }

    fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        #[cfg(all(feature = "cuda", any(test, doctest)))]
        init_gpu();

        let degree = polynomials[0].len();
        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };
        // println!("salt_size: {:?}", salt_size);

        let ret = polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect();
        return ret;
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.get(index);
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let quotient = composition_poly.divide_by_linear(*point);
            // quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }
        // NOTE: circom_compatability
        // Multiply the final polynomial by `X`, so that `final_poly` has the maximum degree for
        // which the LDT will pass. See github.com/mir-protocol/plonky2/pull/436 for details.
        final_poly.coeffs.insert(0, F::Extension::ZERO);

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            timing,
        );

        fri_proof
    }
}
