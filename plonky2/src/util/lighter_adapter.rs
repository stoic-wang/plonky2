//! Adapter for loading lighter-prover JSON format circuits.
//!
//! This module provides functionality to load circuit data from the JSON format
//! used by lighter-prover into okx/plonky2's internal representation.

use core::ops::Range;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::field::goldilocks_field::GoldilocksField;
use crate::field::types::Field;
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
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};

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
pub fn convert_parsed_gate_to_ref(
    parsed: &ParsedGate,
    config: &CircuitConfig,
) -> Result<GateRef<GoldilocksField, 2>> {
    match parsed.gate_type.as_str() {
        "NoopGate" => Ok(GateRef::new(NoopGate)),

        "PublicInputGate" => Ok(GateRef::new(PublicInputGate)),

        "PoseidonGate" => Ok(GateRef::new(PoseidonGate::new())),

        "PoseidonMdsGate" => Ok(GateRef::new(PoseidonMdsGate::new())),

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
            Ok(GateRef::new(ArithmeticExtensionGate::<2> { num_ops }))
        }

        "MulExtensionGate" => {
            let num_ops = get_param(&parsed.params, "num_ops")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| MulExtensionGate::<2>::new_from_config(config).num_ops);
            Ok(GateRef::new(MulExtensionGate::<2> { num_ops }))
        }

        "BaseSumGate" => {
            // BaseSumGate<2> is the default, num_limbs is extracted but the gate is generic over B
            let num_limbs = get_param(&parsed.params, "num_limbs")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(63);
            Ok(GateRef::new(BaseSumGate::<2>::new(num_limbs)))
        }

        "ReducingGate" => {
            let num_coeffs = get_param(&parsed.params, "num_coeffs")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| ReducingGate::<2>::max_coeffs_len(config.num_wires, config.num_routed_wires));
            Ok(GateRef::new(ReducingGate::<2>::new(num_coeffs)))
        }

        "ReducingExtensionGate" => {
            let num_coeffs = get_param(&parsed.params, "num_coeffs")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| ReducingExtensionGate::<2>::max_coeffs_len(config.num_wires, config.num_routed_wires));
            Ok(GateRef::new(ReducingExtensionGate::<2>::new(num_coeffs)))
        }

        "ExponentiationGate" => {
            let num_power_bits = get_param(&parsed.params, "num_power_bits")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| ExponentiationGate::<GoldilocksField, 2>::new_from_config(config).num_power_bits);
            Ok(GateRef::new(ExponentiationGate::<GoldilocksField, 2>::new(num_power_bits)))
        }

        "RandomAccessGate" => {
            let bits = get_param(&parsed.params, "bits")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4);
            // Use new_from_config which is the public constructor
            Ok(GateRef::new(
                RandomAccessGate::<GoldilocksField, 2>::new_from_config(config, bits),
            ))
        }

        "CosetInterpolationGate" => {
            let subgroup_bits = get_param(&parsed.params, "subgroup_bits")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4);
            let degree = get_param(&parsed.params, "degree")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(6);
            // Use with_max_degree which computes barycentric_weights internally
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

/// Load and build CommonCircuitData from a lighter-prover circuit directory.
pub fn load_lighter_common_circuit_data_as_okx<P: AsRef<Path>>(
    dir: P,
) -> Result<CommonCircuitData<GoldilocksField, 2>> {
    let common_path = dir.as_ref().join("common_circuit_data.json");
    let lighter = load_lighter_common_circuit_data(&common_path)?;
    build_common_circuit_data(&lighter)
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
        assert!(gate.params.iter().any(|(k, _)| k == "base"));
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
        let testdata_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("lighter-gnark-plonk-verifier/testdata/step");

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
        let testdata_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("lighter-gnark-plonk-verifier/testdata/step");

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
    fn test_common_circuit_data_serialization() {
        let testdata_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("lighter-gnark-plonk-verifier/testdata/step");

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
                assert!(gate_id.starts_with(expected),
                    "Gate {} should start with {}, got {}", i, expected, gate_id);
            }

            println!("All {} gate types correctly mapped", expected_gate_types.len());
        } else {
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }
}
