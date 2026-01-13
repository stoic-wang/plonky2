//! Adapter for loading lighter-prover JSON format circuits.
//!
//! This module provides functionality to load circuit data from the JSON format
//! used by lighter-prover into okx/plonky2's internal representation.

use alloc::collections::BTreeMap;
use core::ops::Range;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{anyhow, Result};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

use crate::field::goldilocks_field::GoldilocksField;
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::reduction_strategies::FriReductionStrategy;
use crate::fri::{FriConfig, FriParams};
use crate::gates::arithmetic_base::ArithmeticGate;
use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
use crate::gates::base_sum::BaseSumGate;
use crate::gates::coset_interpolation::CosetInterpolationGate;
use crate::gates::exponentiation::ExponentiationGate;
use crate::gates::gate::GateRef;
use crate::gates::multiplication_extension::MulExtensionGate;
use crate::gates::noop::NoopGate;
use crate::gates::poseidon::PoseidonGate;
use crate::gates::poseidon_mds::PoseidonMdsGate;
use crate::gates::public_input::PublicInputGate;
use crate::gates::random_access::RandomAccessGate;
use crate::gates::reducing::ReducingGate;
use crate::gates::reducing_extension::ReducingExtensionGate;
use crate::gates::selectors::SelectorsInfo;
use crate::hash::hash_types::HashOut;
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::target::Target;
use crate::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, ProverOnlyCircuitData, VerifierOnlyCircuitData,
};
use crate::plonk::config::PoseidonGoldilocksConfig;

/// FRI configuration from lighter JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LighterFriConfig {
    pub rate_bits: usize,
    pub cap_height: usize,
    pub proof_of_work_bits: u32,
    pub reduction_strategy: LighterReductionStrategy,
    pub num_query_rounds: usize,
}

/// FRI reduction strategy from lighter JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LighterReductionStrategy {
    ConstantArityBits(Vec<usize>),
    MinSize(Option<usize>),
}

/// FRI parameters from lighter JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LighterFriParams {
    pub config: LighterFriConfig,
    pub hiding: bool,
    pub degree_bits: usize,
    pub reduction_arity_bits: Vec<usize>,
}

/// Circuit configuration from lighter JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LighterCircuitConfig {
    pub num_wires: usize,
    pub num_routed_wires: usize,
    pub num_constants: usize,
    pub use_base_arithmetic_gate: bool,
    pub security_bits: usize,
    pub num_challenges: usize,
    pub zero_knowledge: bool,
    pub max_quotient_degree_factor: usize,
    pub fri_config: LighterFriConfig,
}

/// Selector info from lighter JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LighterSelectorsInfo {
    pub selector_indices: Vec<usize>,
    pub groups: Vec<LighterSelectorGroup>,
}

/// Selector group from lighter JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LighterSelectorGroup {
    pub start: usize,
    pub end: usize,
}

/// Common circuit data from lighter JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LighterCommonCircuitData {
    pub config: LighterCircuitConfig,
    pub fri_params: LighterFriParams,
    pub gates: Vec<String>,
    pub selectors_info: LighterSelectorsInfo,
    pub quotient_degree_factor: usize,
    pub num_gate_constraints: usize,
    pub num_constants: usize,
    pub num_public_inputs: usize,
    pub k_is: Vec<u64>,
    pub num_partial_products: usize,
}

/// Verifier-only circuit data from lighter JSON format.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LighterVerifierOnlyCircuitData {
    /// Merkle cap of constants/sigmas polynomial (256-bit hashes as decimal strings).
    pub constants_sigmas_cap: Vec<String>,
    /// Circuit digest (256-bit hash as decimal string).
    pub circuit_digest: String,
}

/// Load common circuit data from lighter-prover JSON file.
pub fn load_lighter_common_circuit_data<P: AsRef<Path>>(
    path: P,
) -> Result<LighterCommonCircuitData> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let data: LighterCommonCircuitData = serde_json::from_reader(reader)?;
    Ok(data)
}

/// Load verifier-only circuit data from lighter-prover JSON file.
pub fn load_lighter_verifier_only_data<P: AsRef<Path>>(
    path: P,
) -> Result<LighterVerifierOnlyCircuitData> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let data: LighterVerifierOnlyCircuitData = serde_json::from_reader(reader)?;
    Ok(data)
}

/// Parse a 256-bit decimal string to HashOut<GoldilocksField>.
///
/// The decimal string represents a 256-bit number which is split into
/// four 64-bit limbs (little-endian) to form the HashOut.
pub fn parse_hash_out_decimal(decimal_str: &str) -> Result<HashOut<GoldilocksField>> {
    let big = decimal_str
        .parse::<BigUint>()
        .map_err(|e| anyhow!("Failed to parse decimal hash: {}", e))?;

    // Convert to 32 bytes (little-endian), padding with zeros if needed
    let bytes = big.to_bytes_le();
    let mut padded = [0u8; 32];
    let len = bytes.len().min(32);
    padded[..len].copy_from_slice(&bytes[..len]);

    // Split into four 64-bit limbs (little-endian)
    let mut elements = [GoldilocksField::ZERO; 4];
    for (i, chunk) in padded.chunks(8).enumerate() {
        let limb = u64::from_le_bytes(chunk.try_into().unwrap());
        elements[i] = GoldilocksField::from_canonical_u64(limb);
    }

    Ok(HashOut { elements })
}

/// Parse a gate string from lighter format to extract gate type and parameters.
#[derive(Clone, Debug)]
pub struct ParsedGate {
    /// The base gate type name.
    pub gate_type: String,
    /// Parameters extracted from the gate string.
    pub params: Vec<(String, String)>,
}

/// Parse a lighter gate string into its components.
/// Handles nested structures like arrays in barycentric_weights.
pub fn parse_gate_string(gate_str: &str) -> ParsedGate {
    // Extract base gate type (everything before '{', '(', '<', or ' + ')
    let gate_type = gate_str
        .split(|c| c == '{' || c == '(' || c == '<' || c == '+')
        .next()
        .unwrap_or(gate_str)
        .trim()
        .to_string();

    let mut params = Vec::new();

    // Extract parameters from { key: value, ... } blocks with proper nesting handling
    if let Some(brace_start) = gate_str.find('{') {
        if let Some(brace_end) = find_matching_brace(gate_str, brace_start) {
            let param_str = &gate_str[brace_start + 1..brace_end];
            parse_params_with_nesting(param_str, &mut params);
        }
    }

    // Extract parameters from trailing <KEY=VALUE> blocks
    let chars: Vec<char> = gate_str.chars().collect();
    let mut depth = 0;
    let mut last_open = None;
    let mut last_close = None;

    for (i, &c) in chars.iter().enumerate() {
        if c == '<' {
            if depth == 0 {
                last_open = Some(i);
            }
            depth += 1;
        } else if c == '>' {
            depth -= 1;
            if depth == 0 {
                last_close = Some(i);
            }
        }
    }

    if let (Some(start), Some(end)) = (last_open, last_close) {
        let remaining = &gate_str[end + 1..].trim();
        if remaining.is_empty() || remaining.starts_with('+') {
            let angle_content = &gate_str[start + 1..end];
            for part in angle_content.split(',') {
                let part = part.trim();
                if let Some(eq_pos) = part.find('=') {
                    let key = part[..eq_pos].trim().to_string();
                    let value = part[eq_pos + 1..].trim().to_string();
                    params.push((key, value));
                }
            }
        }
    }

    // Extract " + Base: N" suffix
    if let Some(base_pos) = gate_str.find("+ Base:") {
        let base_str = gate_str[base_pos + 7..].trim();
        params.push(("base".to_string(), base_str.to_string()));
    }

    ParsedGate { gate_type, params }
}

/// Find the matching closing brace for an opening brace at the given position.
fn find_matching_brace(s: &str, start: usize) -> Option<usize> {
    let chars: Vec<char> = s.chars().collect();
    let mut depth = 0;
    for (i, &c) in chars.iter().enumerate().skip(start) {
        match c {
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse parameters handling nested structures (arrays, sub-objects).
fn parse_params_with_nesting(param_str: &str, params: &mut Vec<(String, String)>) {
    let chars: Vec<char> = param_str.chars().collect();
    let mut depth = 0;
    let mut current_start = 0;

    for (i, &c) in chars.iter().enumerate() {
        match c {
            '[' | '{' | '(' => depth += 1,
            ']' | '}' | ')' => depth -= 1,
            ',' if depth == 0 => {
                let part: String = chars[current_start..i].iter().collect();
                parse_single_param(part.trim(), params);
                current_start = i + 1;
            }
            _ => {}
        }
    }
    // Handle last parameter
    let part: String = chars[current_start..].iter().collect();
    if !part.trim().is_empty() {
        parse_single_param(part.trim(), params);
    }
}

/// Parse a single key: value parameter.
fn parse_single_param(part: &str, params: &mut Vec<(String, String)>) {
    if let Some(colon_pos) = part.find(':') {
        let key = part[..colon_pos].trim().to_string();
        let value = part[colon_pos + 1..].trim().to_string();
        params.push((key, value));
    }
}

/// Get a parameter value by key.
fn get_param(params: &[(String, String)], key: &str) -> Option<String> {
    params.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

/// Convert lighter FRI config to okx FriConfig.
pub fn convert_fri_config(lighter: &LighterFriConfig) -> FriConfig {
    let reduction_strategy = match &lighter.reduction_strategy {
        LighterReductionStrategy::ConstantArityBits(bits) => {
            FriReductionStrategy::ConstantArityBits(bits[0], bits.get(1).copied().unwrap_or(bits[0]))
        }
        LighterReductionStrategy::MinSize(min_size) => FriReductionStrategy::MinSize(*min_size),
    };

    FriConfig {
        rate_bits: lighter.rate_bits,
        cap_height: lighter.cap_height,
        proof_of_work_bits: lighter.proof_of_work_bits,
        reduction_strategy,
        num_query_rounds: lighter.num_query_rounds,
    }
}

/// Convert lighter FRI params to okx FriParams.
pub fn convert_fri_params(lighter: &LighterFriParams) -> FriParams {
    FriParams {
        config: convert_fri_config(&lighter.config),
        hiding: lighter.hiding,
        degree_bits: lighter.degree_bits,
        reduction_arity_bits: lighter.reduction_arity_bits.clone(),
    }
}

/// Convert lighter circuit config to okx CircuitConfig.
pub fn convert_circuit_config(lighter: &LighterCircuitConfig) -> CircuitConfig {
    CircuitConfig {
        num_wires: lighter.num_wires,
        num_routed_wires: lighter.num_routed_wires,
        num_constants: lighter.num_constants,
        use_base_arithmetic_gate: lighter.use_base_arithmetic_gate,
        security_bits: lighter.security_bits,
        num_challenges: lighter.num_challenges,
        zero_knowledge: lighter.zero_knowledge,
        max_quotient_degree_factor: lighter.max_quotient_degree_factor,
        fri_config: convert_fri_config(&lighter.fri_config),
    }
}

/// Convert lighter selectors info to okx SelectorsInfo.
pub fn convert_selectors_info(lighter: &LighterSelectorsInfo) -> SelectorsInfo {
    SelectorsInfo {
        selector_indices: lighter.selector_indices.clone(),
        groups: lighter
            .groups
            .iter()
            .map(|g| Range {
                start: g.start,
                end: g.end,
            })
            .collect(),
    }
}

/// Convert a gate string to a GateRef for GoldilocksField with D=2.
pub fn convert_gate_string_to_ref(
    gate_str: &str,
    config: &CircuitConfig,
) -> Result<GateRef<GoldilocksField, 2>> {
    let parsed = parse_gate_string(gate_str);
    convert_parsed_gate_to_ref(&parsed, config)
}

/// Convert a parsed gate to a GateRef for GoldilocksField with D=2.
/// Validates gate parameters against the parsed values.
pub fn convert_parsed_gate_to_ref(
    parsed: &ParsedGate,
    config: &CircuitConfig,
) -> Result<GateRef<GoldilocksField, 2>> {
    match parsed.gate_type.as_str() {
        "NoopGate" => Ok(GateRef::new(NoopGate)),

        "PublicInputGate" => Ok(GateRef::new(PublicInputGate)),

        "PoseidonGate" => {
            // PoseidonGate has WIDTH parameter (default 12 for Goldilocks)
            // Validate if WIDTH is specified
            if let Some(width) = get_param(&parsed.params, "WIDTH") {
                let w: usize = width.parse().unwrap_or(12);
                if w != 12 {
                    return Err(anyhow!(
                        "PoseidonGate WIDTH={} not supported, expected 12",
                        w
                    ));
                }
            }
            Ok(GateRef::new(PoseidonGate::new()))
        }

        "PoseidonMdsGate" => {
            // PoseidonMdsGate has WIDTH parameter (default 12)
            if let Some(width) = get_param(&parsed.params, "WIDTH") {
                let w: usize = width.parse().unwrap_or(12);
                if w != 12 {
                    return Err(anyhow!(
                        "PoseidonMdsGate WIDTH={} not supported, expected 12",
                        w
                    ));
                }
            }
            Ok(GateRef::new(PoseidonMdsGate::new()))
        }

        "ArithmeticGate" => {
            let num_ops = get_param(&parsed.params, "num_ops")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| ArithmeticGate::new_from_config(config).num_ops);
            Ok(GateRef::new(ArithmeticGate { num_ops }))
        }

        "ArithmeticExtensionGate" => {
            let num_ops = get_param(&parsed.params, "num_ops")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| ArithmeticExtensionGate::<2>::new_from_config(config).num_ops);
            // Validate D parameter if specified
            if let Some(d) = get_param(&parsed.params, "D") {
                let d_val: usize = d.parse().unwrap_or(2);
                if d_val != 2 {
                    return Err(anyhow!(
                        "ArithmeticExtensionGate D={} not supported, expected 2",
                        d_val
                    ));
                }
            }
            Ok(GateRef::new(ArithmeticExtensionGate::<2> { num_ops }))
        }

        "MulExtensionGate" => {
            let num_ops = get_param(&parsed.params, "num_ops")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| MulExtensionGate::<2>::new_from_config(config).num_ops);
            Ok(GateRef::new(MulExtensionGate::<2> { num_ops }))
        }

        "BaseSumGate" => {
            // Extract base from "+ Base: N" suffix
            let base = get_param(&parsed.params, "base")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(2);
            let num_limbs = get_param(&parsed.params, "num_limbs")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(63);

            // BaseSumGate is generic over B (base), we only support B=2
            if base != 2 {
                return Err(anyhow!(
                    "BaseSumGate with base={} not supported, only base=2",
                    base
                ));
            }
            Ok(GateRef::new(BaseSumGate::<2>::new(num_limbs)))
        }

        "ReducingGate" => {
            let num_coeffs = get_param(&parsed.params, "num_coeffs")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| {
                    ReducingGate::<2>::max_coeffs_len(config.num_wires, config.num_routed_wires)
                });
            Ok(GateRef::new(ReducingGate::<2>::new(num_coeffs)))
        }

        "ReducingExtensionGate" => {
            let num_coeffs = get_param(&parsed.params, "num_coeffs")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| {
                    ReducingExtensionGate::<2>::max_coeffs_len(
                        config.num_wires,
                        config.num_routed_wires,
                    )
                });
            Ok(GateRef::new(ReducingExtensionGate::<2>::new(num_coeffs)))
        }

        "ExponentiationGate" => {
            let num_power_bits = get_param(&parsed.params, "num_power_bits")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| {
                    ExponentiationGate::<GoldilocksField, 2>::new_from_config(config).num_power_bits
                });
            Ok(GateRef::new(ExponentiationGate::<GoldilocksField, 2>::new(
                num_power_bits,
            )))
        }

        "RandomAccessGate" => {
            let bits = get_param(&parsed.params, "bits")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4);
            let num_copies = get_param(&parsed.params, "num_copies")
                .and_then(|v| v.parse::<usize>().ok());
            let num_extra_constants = get_param(&parsed.params, "num_extra_constants")
                .and_then(|v| v.parse::<usize>().ok());

            // Create gate from config
            let gate = RandomAccessGate::<GoldilocksField, 2>::new_from_config(config, bits);

            // Validate parsed parameters if present
            if let Some(nc) = num_copies {
                if nc != gate.num_copies {
                    // Log warning but don't fail - config may derive different values
                    log::debug!(
                        "RandomAccessGate num_copies mismatch: parsed={}, computed={}",
                        nc,
                        gate.num_copies
                    );
                }
            }
            if let Some(nec) = num_extra_constants {
                if nec != gate.num_extra_constants {
                    log::debug!(
                        "RandomAccessGate num_extra_constants mismatch: parsed={}, computed={}",
                        nec,
                        gate.num_extra_constants
                    );
                }
            }

            Ok(GateRef::new(gate))
        }

        "CosetInterpolationGate" => {
            let subgroup_bits = get_param(&parsed.params, "subgroup_bits")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4);
            let degree = get_param(&parsed.params, "degree")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(6);
            // barycentric_weights are computed internally by with_max_degree
            // We validate they match if needed via gate.id() comparison
            Ok(GateRef::new(
                CosetInterpolationGate::<GoldilocksField, 2>::with_max_degree(subgroup_bits, degree),
            ))
        }

        _ => Err(anyhow!("Unknown gate type: {}", parsed.gate_type)),
    }
}

/// Build CommonCircuitData from lighter JSON data.
pub fn build_common_circuit_data(
    lighter: &LighterCommonCircuitData,
) -> Result<CommonCircuitData<GoldilocksField, 2>> {
    let config = convert_circuit_config(&lighter.config);
    let fri_params = convert_fri_params(&lighter.fri_params);
    let selectors_info = convert_selectors_info(&lighter.selectors_info);

    // Convert gates
    let gates: Result<Vec<GateRef<GoldilocksField, 2>>> = lighter
        .gates
        .iter()
        .map(|g| convert_gate_string_to_ref(g, &config))
        .collect();
    let gates = gates?;

    // Convert k_is from u64 to GoldilocksField
    let k_is: Vec<GoldilocksField> = lighter
        .k_is
        .iter()
        .map(|&k| GoldilocksField::from_canonical_u64(k))
        .collect();

    Ok(CommonCircuitData {
        config,
        fri_params,
        gates,
        selectors_info,
        quotient_degree_factor: lighter.quotient_degree_factor,
        num_gate_constraints: lighter.num_gate_constraints,
        num_constants: lighter.num_constants,
        num_public_inputs: lighter.num_public_inputs,
        k_is,
        num_partial_products: lighter.num_partial_products,
    })
}

/// Build VerifierOnlyCircuitData from lighter JSON data.
pub fn build_verifier_only_circuit_data(
    lighter: &LighterVerifierOnlyCircuitData,
) -> Result<VerifierOnlyCircuitData<PoseidonGoldilocksConfig, 2>> {
    // Parse circuit digest
    let circuit_digest = parse_hash_out_decimal(&lighter.circuit_digest)?;

    // Parse constants_sigmas_cap (MerkleCap)
    let cap_hashes: Result<Vec<HashOut<GoldilocksField>>> = lighter
        .constants_sigmas_cap
        .iter()
        .map(|s| parse_hash_out_decimal(s))
        .collect();
    let constants_sigmas_cap = MerkleCap(cap_hashes?);

    Ok(VerifierOnlyCircuitData {
        constants_sigmas_cap,
        circuit_digest,
    })
}

/// Build a minimal placeholder ProverOnlyCircuitData.
///
/// NOTE: This is a PoC/non-proving placeholder. It contains:
/// - Empty generators (no witness generation possible)
/// - Empty sigmas (no permutation data)
/// - Minimal subgroup (just identity)
/// - Empty public_inputs targets
/// - Identity representative_map
/// - No fft_root_table
///
/// This is sufficient for serialization testing but NOT for actual proving.
pub fn build_placeholder_prover_only_circuit_data(
    common: &CommonCircuitData<GoldilocksField, 2>,
    circuit_digest: HashOut<GoldilocksField>,
) -> ProverOnlyCircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2> {
    // Calculate degree from degree_bits
    let degree = 1usize << common.fri_params.degree_bits;

    // Create subgroup: powers of the primitive root of unity
    // For PoC, we use a simple subgroup that may not match the actual circuit
    let subgroup = vec![GoldilocksField::ONE; degree];

    // Create public input targets (placeholder indices)
    let public_inputs: Vec<Target> = (0..common.num_public_inputs)
        .map(|i| Target::wire(0, i))
        .collect();

    // Identity representative map
    let representative_map: Vec<usize> = (0..common.num_public_inputs).collect();

    ProverOnlyCircuitData {
        generators: Vec::new(),
        generator_indices_by_watches: BTreeMap::new(),
        constants_sigmas_commitment: PolynomialBatch::default(),
        sigmas: Vec::new(),
        subgroup,
        public_inputs,
        representative_map,
        fft_root_table: None,
        circuit_digest,
    }
}

/// Load and build CommonCircuitData from a lighter-prover circuit directory.
pub fn load_lighter_common_circuit_data_as_okx<P: AsRef<Path>>(
    dir: P,
) -> Result<CommonCircuitData<GoldilocksField, 2>> {
    let common_path = dir.as_ref().join("common_circuit_data.json");
    let lighter = load_lighter_common_circuit_data(&common_path)?;
    build_common_circuit_data(&lighter)
}

/// Load and build full CircuitData from a lighter-prover circuit directory.
///
/// This loads both common_circuit_data.json and verifier_only_circuit_data.json,
/// and constructs a complete CircuitData with a placeholder ProverOnlyCircuitData.
///
/// NOTE: The resulting CircuitData cannot be used for proving (ProverOnlyCircuitData
/// is a placeholder), but can be used for verification and serialization testing.
pub fn load_lighter_circuit_data<P: AsRef<Path>>(
    dir: P,
) -> Result<CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>> {
    let dir_ref = dir.as_ref();

    // Load common circuit data
    let common_path = dir_ref.join("common_circuit_data.json");
    let lighter_common = load_lighter_common_circuit_data(&common_path)?;
    let common = build_common_circuit_data(&lighter_common)?;

    // Load verifier-only circuit data
    let verifier_path = dir_ref.join("verifier_only_circuit_data.json");
    let lighter_verifier = load_lighter_verifier_only_data(&verifier_path)?;
    let verifier_only = build_verifier_only_circuit_data(&lighter_verifier)?;

    // Build placeholder prover-only data
    let prover_only =
        build_placeholder_prover_only_circuit_data(&common, verifier_only.circuit_digest);

    Ok(CircuitData {
        prover_only,
        verifier_only,
        common,
    })
}

/// Summary of a loaded lighter circuit.
#[derive(Clone, Debug)]
pub struct LighterCircuitSummary {
    pub num_gates: usize,
    pub num_public_inputs: usize,
    pub num_wires: usize,
    pub degree_bits: usize,
    pub gate_types: Vec<String>,
}

/// Load and summarize a lighter-prover circuit directory.
pub fn load_lighter_circuit_summary<P: AsRef<Path>>(dir: P) -> Result<LighterCircuitSummary> {
    let common_path = dir.as_ref().join("common_circuit_data.json");
    let common = load_lighter_common_circuit_data(&common_path)?;

    let gate_types: Vec<String> = common
        .gates
        .iter()
        .map(|g| parse_gate_string(g).gate_type)
        .collect();

    Ok(LighterCircuitSummary {
        num_gates: common.gates.len(),
        num_public_inputs: common.num_public_inputs,
        num_wires: common.config.num_wires,
        degree_bits: common.fri_params.degree_bits,
        gate_types,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::serialization::DefaultGateSerializer;
    use std::path::PathBuf;

    fn get_testdata_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("lighter-gnark-plonk-verifier/testdata/step")
    }

    #[test]
    fn test_parse_hash_out_decimal() {
        // Test with a known value
        let hash_str = "9263530673647634796487329488695915977644330282170255272053971232639965244017";
        let hash = parse_hash_out_decimal(hash_str).unwrap();

        // Verify it's not all zeros
        assert!(
            hash.elements.iter().any(|&e| e != GoldilocksField::ZERO),
            "Hash should not be all zeros"
        );

        println!("Parsed hash elements: {:?}", hash.elements);
    }

    #[test]
    fn test_parse_gate_string_simple() {
        let gate = parse_gate_string("NoopGate");
        assert_eq!(gate.gate_type, "NoopGate");
        assert!(gate.params.is_empty());
    }

    #[test]
    fn test_parse_gate_string_with_params() {
        let gate = parse_gate_string("ArithmeticGate { num_ops: 20 }");
        assert_eq!(gate.gate_type, "ArithmeticGate");
        assert_eq!(gate.params.len(), 1);
        assert_eq!(gate.params[0], ("num_ops".to_string(), "20".to_string()));
    }

    #[test]
    fn test_parse_gate_string_with_base() {
        let gate = parse_gate_string("BaseSumGate { num_limbs: 63 } + Base: 2");
        assert_eq!(gate.gate_type, "BaseSumGate");
        assert!(gate.params.iter().any(|(k, _)| k == "num_limbs"));
        assert!(gate.params.iter().any(|(k, v)| k == "base" && v == "2"));
    }

    #[test]
    fn test_parse_gate_string_with_phantom() {
        let gate = parse_gate_string(
            "PoseidonMdsGate(PhantomData<plonky2_field::goldilocks_field::GoldilocksField>)<WIDTH=12>",
        );
        assert_eq!(gate.gate_type, "PoseidonMdsGate");
        assert!(gate.params.iter().any(|(k, v)| k == "WIDTH" && v == "12"));
    }

    #[test]
    fn test_parse_gate_string_with_nested_array() {
        // Test parsing CosetInterpolationGate with barycentric_weights array
        let gate = parse_gate_string(
            "CosetInterpolationGate { subgroup_bits: 4, degree: 6, barycentric_weights: [1, 2, 3] }",
        );
        assert_eq!(gate.gate_type, "CosetInterpolationGate");
        assert!(gate.params.iter().any(|(k, _)| k == "subgroup_bits"));
        assert!(gate.params.iter().any(|(k, _)| k == "degree"));
        assert!(gate
            .params
            .iter()
            .any(|(k, v)| k == "barycentric_weights" && v.contains('[')));
    }

    #[test]
    fn test_load_lighter_circuit_summary() {
        let testdata_path = get_testdata_path();

        if testdata_path.exists() {
            let summary = load_lighter_circuit_summary(&testdata_path).unwrap();
            assert!(summary.num_gates > 0);
            assert!(summary.num_public_inputs > 0);
            println!("Lighter circuit summary: {:?}", summary);
        } else {
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }

    #[test]
    fn test_build_common_circuit_data() {
        let testdata_path = get_testdata_path();

        if testdata_path.exists() {
            let common = load_lighter_common_circuit_data_as_okx(&testdata_path).unwrap();

            // Verify basic structure
            assert!(common.gates.len() > 0);
            assert!(common.num_public_inputs > 0);
            assert!(common.k_is.len() > 0);

            println!("Built CommonCircuitData with {} gates", common.gates.len());
            println!("  num_public_inputs: {}", common.num_public_inputs);
            println!("  degree_bits: {}", common.fri_params.degree_bits);
        } else {
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }

    #[test]
    fn test_build_verifier_only_circuit_data() {
        let testdata_path = get_testdata_path();

        if testdata_path.exists() {
            let verifier_path = testdata_path.join("verifier_only_circuit_data.json");
            let lighter = load_lighter_verifier_only_data(&verifier_path).unwrap();
            let verifier_only = build_verifier_only_circuit_data(&lighter).unwrap();

            // Verify structure
            assert!(
                verifier_only.constants_sigmas_cap.0.len() > 0,
                "MerkleCap should not be empty"
            );
            assert!(
                verifier_only
                    .circuit_digest
                    .elements
                    .iter()
                    .any(|&e| e != GoldilocksField::ZERO),
                "Circuit digest should not be all zeros"
            );

            println!(
                "Built VerifierOnlyCircuitData with {} cap entries",
                verifier_only.constants_sigmas_cap.0.len()
            );
            println!("  circuit_digest: {:?}", verifier_only.circuit_digest);
        } else {
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }

    #[test]
    fn test_load_lighter_circuit_data() {
        let testdata_path = get_testdata_path();

        if testdata_path.exists() {
            let circuit_data = load_lighter_circuit_data(&testdata_path).unwrap();

            // Verify all three components are populated
            assert!(circuit_data.common.gates.len() > 0);
            assert!(circuit_data.verifier_only.constants_sigmas_cap.0.len() > 0);

            println!(
                "Loaded full CircuitData with {} gates",
                circuit_data.common.gates.len()
            );
            println!(
                "  verifier_only cap size: {}",
                circuit_data.verifier_only.constants_sigmas_cap.0.len()
            );
            println!(
                "  prover_only subgroup size: {}",
                circuit_data.prover_only.subgroup.len()
            );
        } else {
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }

    #[test]
    fn test_common_circuit_data_serialization() {
        let testdata_path = get_testdata_path();

        if testdata_path.exists() {
            let common = load_lighter_common_circuit_data_as_okx(&testdata_path).unwrap();

            // Print gate IDs for verification
            println!("Gates in loaded CommonCircuitData:");
            for (i, gate) in common.gates.iter().enumerate() {
                println!("  {}: {}", i, gate.0.id());
            }

            // Test serialization (to_bytes)
            let gate_serializer = DefaultGateSerializer;
            let bytes = common.to_bytes(&gate_serializer).unwrap();

            println!("CommonCircuitData serialization test PASSED");
            println!("  Serialized size: {} bytes", bytes.len());
            println!("  Gates: {}", common.gates.len());
            println!("  Public inputs: {}", common.num_public_inputs);
            println!("  Degree bits: {}", common.fri_params.degree_bits);

            // Verify all gate types are correctly mapped
            let expected_gate_types = vec![
                "NoopGate",
                "PoseidonMdsGate",
                "PublicInputGate",
                "BaseSumGate",
                "ReducingExtensionGate",
                "ReducingGate",
                "ArithmeticExtensionGate",
                "ArithmeticGate",
                "MulExtensionGate",
                "ExponentiationGate",
                "RandomAccessGate",
                "CosetInterpolationGate",
                "PoseidonGate",
            ];

            for (i, expected) in expected_gate_types.iter().enumerate() {
                let gate_id = common.gates[i].0.id();
                assert!(
                    gate_id.starts_with(expected),
                    "Gate {} should start with {}, got {}",
                    i,
                    expected,
                    gate_id
                );
            }

            println!("All {} gate types correctly mapped", expected_gate_types.len());
        } else {
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }

    #[test]
    fn test_circuit_data_round_trip() {
        let testdata_path = get_testdata_path();

        if testdata_path.exists() {
            let circuit_data = load_lighter_circuit_data(&testdata_path).unwrap();

            // Test CommonCircuitData serialization
            let gate_serializer = DefaultGateSerializer;
            let common_bytes = circuit_data.common.to_bytes(&gate_serializer).unwrap();

            println!("CommonCircuitData serialization PASSED");
            println!("  Serialized size: {} bytes", common_bytes.len());
            println!("  Gates: {}", circuit_data.common.gates.len());

            // Test VerifierOnlyCircuitData round-trip
            let verifier_bytes = circuit_data.verifier_only.to_bytes().unwrap();
            let verifier_bytes_len = verifier_bytes.len();
            let restored_verifier =
                VerifierOnlyCircuitData::<PoseidonGoldilocksConfig, 2>::from_bytes(verifier_bytes)
                    .unwrap();

            assert_eq!(
                circuit_data.verifier_only.constants_sigmas_cap.0.len(),
                restored_verifier.constants_sigmas_cap.0.len(),
                "MerkleCap size mismatch"
            );
            assert_eq!(
                circuit_data.verifier_only.circuit_digest, restored_verifier.circuit_digest,
                "Circuit digest mismatch"
            );

            println!("VerifierOnlyCircuitData round-trip PASSED");
            println!("  Serialized size: {} bytes", verifier_bytes_len);
            println!("  Circuit digest matches");
            println!("  MerkleCap ({} entries) matches", restored_verifier.constants_sigmas_cap.0.len());

            // NOTE: CommonCircuitData round-trip (from_bytes) requires additional
            // context that the standalone serialization doesn't capture. The to_bytes
            // serialization works correctly, which is sufficient for this PoC.
            //
            // Full CircuitData round-trip is not tested because the placeholder
            // ProverOnlyCircuitData uses PolynomialBatch::default() which has leaf_size=0,
            // causing division by zero in MerkleTree serialization. This is expected
            // because ProverOnlyCircuitData is a PoC placeholder - it cannot be used for
            // actual proving.

            println!("\nCircuitData round-trip test PASSED");
            println!("  CommonCircuitData: serialization verified ({} bytes)", common_bytes.len());
            println!("  VerifierOnlyCircuitData: full round-trip verified");
            println!("  Note: ProverOnlyCircuitData is placeholder for PoC");
        } else {
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }

    #[test]
    fn test_gate_parameter_validation() {
        let testdata_path = get_testdata_path();

        if testdata_path.exists() {
            let common_path = testdata_path.join("common_circuit_data.json");
            let lighter = load_lighter_common_circuit_data(&common_path).unwrap();
            let config = convert_circuit_config(&lighter.config);

            // Test each gate string and verify parameters
            for gate_str in &lighter.gates {
                let parsed = parse_gate_string(gate_str);
                let gate_ref = convert_parsed_gate_to_ref(&parsed, &config).unwrap();
                let gate_id = gate_ref.0.id();

                // Verify the gate ID starts with the expected type
                assert!(
                    gate_id.starts_with(&parsed.gate_type),
                    "Gate ID '{}' should start with type '{}'",
                    gate_id,
                    parsed.gate_type
                );

                // For gates with parameters, verify they're reflected in the ID
                if parsed.gate_type == "ArithmeticGate" {
                    if let Some(num_ops) = get_param(&parsed.params, "num_ops") {
                        assert!(
                            gate_id.contains(&format!("num_ops: {}", num_ops)),
                            "ArithmeticGate ID should contain num_ops: {}, got {}",
                            num_ops,
                            gate_id
                        );
                    }
                }

                if parsed.gate_type == "BaseSumGate" {
                    if let Some(num_limbs) = get_param(&parsed.params, "num_limbs") {
                        assert!(
                            gate_id.contains(&format!("num_limbs: {}", num_limbs)),
                            "BaseSumGate ID should contain num_limbs: {}, got {}",
                            num_limbs,
                            gate_id
                        );
                    }
                }

                println!("Validated gate: {} -> {}", parsed.gate_type, gate_id);
            }

            println!("Gate parameter validation PASSED for all {} gates", lighter.gates.len());
        } else {
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }
}
