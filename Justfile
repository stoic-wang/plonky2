# Justfile for okx/plonky2 builds
# Usage: just <recipe>

set shell := ["bash", "-c"]

# Path to zkvm-build conda environment (for libclang)
zkvm_env := "/scratch/bangyanwang/miniconda3/envs/zkvm-build"
cuda_home := "/usr/local/cuda-12.8"

# Default recipe
default:
    @just --list

# ============================================
# GPU Builds (require libclang + CUDA)
# ============================================

# Build plonky2_field with CUDA support
build-gpu:
    @echo "Building with CUDA support..."
    PATH={{cuda_home}}/bin:$PATH \
    CUDA_HOME={{cuda_home}} \
    LIBCLANG_PATH={{zkvm_env}}/lib \
    LD_LIBRARY_PATH={{cuda_home}}/lib64:{{zkvm_env}}/lib:$LD_LIBRARY_PATH \
    BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-redhat-linux/11/include" \
    cargo build --release -p plonky2_field --features=cuda

# Build entire plonky2 workspace with CUDA
build-gpu-all:
    @echo "Building entire workspace with CUDA support..."
    PATH={{cuda_home}}/bin:$PATH \
    CUDA_HOME={{cuda_home}} \
    LIBCLANG_PATH={{zkvm_env}}/lib \
    LD_LIBRARY_PATH={{cuda_home}}/lib64:{{zkvm_env}}/lib:$LD_LIBRARY_PATH \
    BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-redhat-linux/11/include" \
    cargo build --release --features=cuda

# Run FFT tests with GPU (quick verification)
test-gpu:
    @echo "Running FFT tests with CUDA..."
    PATH={{cuda_home}}/bin:$PATH \
    CUDA_HOME={{cuda_home}} \
    LIBCLANG_PATH={{zkvm_env}}/lib \
    LD_LIBRARY_PATH={{cuda_home}}/lib64:{{zkvm_env}}/lib:$LD_LIBRARY_PATH \
    BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-redhat-linux/11/include" \
    cargo test --release -p plonky2_field --features=cuda --lib -- fft --nocapture

# Run ECDSA example with GPU (slower, full proof)
run-ecdsa-gpu:
    @echo "Running ECDSA with CUDA..."
    PATH={{cuda_home}}/bin:$PATH \
    CUDA_HOME={{cuda_home}} \
    LIBCLANG_PATH={{zkvm_env}}/lib \
    LD_LIBRARY_PATH={{cuda_home}}/lib64:{{zkvm_env}}/lib:$LD_LIBRARY_PATH \
    BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-redhat-linux/11/include" \
    cargo run --release -p ecdsa --features=cuda --example ecdsa_secp256k1

# ============================================
# CPU Builds (no special environment needed)
# ============================================

# Build with CPU only (no CUDA)
build-cpu:
    cargo build --release -p plonky2 --features=no_cuda

# Build entire workspace CPU only
build-cpu-all:
    cargo build --release --features=no_cuda

# Run ECDSA example (CPU)
run-ecdsa:
    cargo run --release -p ecdsa --example ecdsa_secp256k1

# Run all tests (CPU)
test:
    cargo test --release

# ============================================
# Benchmarks
# ============================================

# FFT benchmark (CPU)
bench-fft:
    cargo bench --release -p plonky2_field -- fft

# FFT benchmark (GPU)
bench-fft-gpu:
    PATH={{cuda_home}}/bin:$PATH \
    CUDA_HOME={{cuda_home}} \
    LIBCLANG_PATH={{zkvm_env}}/lib \
    LD_LIBRARY_PATH={{cuda_home}}/lib64:{{zkvm_env}}/lib:$LD_LIBRARY_PATH \
    BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-redhat-linux/11/include" \
    cargo bench --release -p plonky2_field --features=cuda -- fft

# ============================================
# Utilities
# ============================================

# Check if libclang is available
check-env:
    @echo "Checking environment..."
    @if [ -f "{{zkvm_env}}/lib/libclang.so" ]; then \
        echo "  libclang: OK ({{zkvm_env}}/lib/libclang.so)"; \
    else \
        echo "  libclang: NOT FOUND"; \
    fi
    @if command -v nvcc &>/dev/null; then \
        echo "  CUDA: $(nvcc --version | grep release | awk '{print $6}')"; \
    else \
        echo "  CUDA: NOT FOUND"; \
    fi
    @echo "  Rust: $(rustc --version)"

# Clean build artifacts
clean:
    cargo clean
