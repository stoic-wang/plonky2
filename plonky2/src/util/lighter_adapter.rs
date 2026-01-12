//! Adapter for loading lighter-prover JSON format circuits.
//!
//! This module provides functionality to load circuit data from the JSON format
//! used by lighter-prover into okx/plonky2's internal representation.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::fri::reduction_strategies::FriReductionStrategy;
use crate::fri::FriConfig;
use crate::plonk::circuit_data::CircuitConfig;

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
///
/// Gate strings are in Rust debug format, e.g.:
/// - "NoopGate"
/// - "ArithmeticGate { num_ops: 20 }"
/// - "PoseidonMdsGate(PhantomData<...>)<WIDTH=12>"
/// - "BaseSumGate { num_limbs: 63 } + Base: 2"
#[derive(Clone, Debug)]
pub struct ParsedGate {
    /// The base gate type name.
    pub gate_type: String,
    /// Parameters extracted from the gate string.
    pub params: Vec<(String, String)>,
}

/// Parse a lighter gate string into its components.
pub fn parse_gate_string(gate_str: &str) -> ParsedGate {
    // Extract base gate type (everything before '{', '(', '<', or ' + ')
    let gate_type = gate_str
        .split(|c| c == '{' || c == '(' || c == '<' || c == '+')
        .next()
        .unwrap_or(gate_str)
        .trim()
        .to_string();

    let mut params = Vec::new();

    // Extract parameters from { key: value } blocks
    if let Some(start) = gate_str.find('{') {
        if let Some(end) = gate_str.find('}') {
            let param_str = &gate_str[start + 1..end];
            for part in param_str.split(',') {
                let part = part.trim();
                if let Some(colon_pos) = part.find(':') {
                    let key = part[..colon_pos].trim().to_string();
                    let value = part[colon_pos + 1..].trim().to_string();
                    params.push((key, value));
                }
            }
        }
    }

    // Extract parameters from trailing <KEY=VALUE> blocks (after last '>')
    // This handles cases like "Gate(PhantomData<...>)<WIDTH=12>"
    // We want to find the trailing <...> block, not the PhantomData<...> block
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

    // If there's a trailing <...> block at the end of the string
    if let (Some(start), Some(end)) = (last_open, last_close) {
        // Check if this is truly at the end (allow trailing whitespace)
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

/// Convert lighter FRI config to okx FriConfig.
pub fn convert_fri_config(lighter: &LighterFriConfig) -> FriConfig {
    let reduction_strategy = match &lighter.reduction_strategy {
        LighterReductionStrategy::ConstantArityBits(bits) => {
            FriReductionStrategy::ConstantArityBits(bits[0], bits.get(1).copied().unwrap_or(bits[0]))
        }
        LighterReductionStrategy::MinSize(min_size) => {
            FriReductionStrategy::MinSize(*min_size)
        }
    };

    FriConfig {
        rate_bits: lighter.rate_bits,
        cap_height: lighter.cap_height,
        proof_of_work_bits: lighter.proof_of_work_bits,
        reduction_strategy,
        num_query_rounds: lighter.num_query_rounds,
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
    fn test_load_lighter_circuit_summary() {
        // Try to load the testdata if available
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
            // Skip if testdata not available
            println!("Testdata not found at {:?}, skipping", testdata_path);
        }
    }
}
