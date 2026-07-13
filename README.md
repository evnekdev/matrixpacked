# matrixpacked

Packed triangular, symmetric, positive-definite, and Hermitian matrices backed directly by traditional BLAS/LAPACK packed-column storage.

## Backend

The crate contains wrappers but does not force one native BLAS/LAPACK implementation. Either link your application's existing provider, or enable the bundled option:

```toml
matrixpacked = { version = "0.1", features = ["openblas-static"] }
```

On Windows, use the bundled static Intel MKL provider:

```powershell
cargo run --example lapack_lower_f64_tpmv --features intel-mkl-static
```

## Packed operations

```rust
use matrixpacked::PackedLower;

let a = PackedLower::<f64>::from_vec(
    3,
    vec![2.0, 3.0, 1.0, 1.0, 4.0, 5.0],
)?;

let y = a.mul_vector(&[1.0, 2.0, 3.0])?;
let x = a.solve_vector(&y)?;
```

For allocation-sensitive code, use caller-owned output and destructive factorization:

```rust
use matrixpacked::PackedSPDViewMut;

let mut ap = [4.0_f64, 1.0, 1.0, 3.0, 0.0, 2.0];
let matrix = PackedSPDViewMut::from_slice_mut(3, &mut ap)?;
let factor = matrix.cholesky_in_place()?; // reuses `ap`

let mut rhs = [9.0, 7.0, 7.0];
factor.solve_vector_in_place(&mut rhs)?;
```

See `OPERATIONS.md` for the supported nalgebra-like operator surface and deliberate structural restrictions.

## LAPACK operation examples

The `examples/` directory contains 105 individually runnable, assertion-based LAPACK examples covering every packed LAPACK operation currently exposed for every supported scalar and matrix family.

For example:

```bash
cargo run --example lapack_lower_f64_tpmv --features openblas-static
cargo run --example lapack_spd_c64_pptrs --features openblas-static
cargo run --example lapack_hermitian_c32_hptri --features openblas-static
```

Run the complete example suite with a bundled OpenBLAS provider:

```bash
./scripts_run_lapack_examples.sh --openblas-static
```

On Windows, run the batch scripts with Intel MKL:

```bat
scripts_run_lapack_examples.bat --intel-mkl-static
scripts_run_nonlapack_examples.bat --intel-mkl-static
```

See [`EXAMPLE_COVERAGE.md`](EXAMPLE_COVERAGE.md) for the full matrix of examples.

## Non-LAPACK examples

Alongside the individual packed BLAS/LAPACK examples, the crate includes detailed examples for operations performed directly on packed storage:

```bash
cargo run --example nonlapack_lower
cargo run --example nonlapack_upper
cargo run --example nonlapack_symmetric
cargo run --example nonlapack_spd
cargo run --example nonlapack_hermitian
```

Run all of them with:

```bash
./scripts_run_nonlapack_examples.sh
```

See `NON_LAPACK_EXAMPLES.md` for the operation-by-operation coverage table.
