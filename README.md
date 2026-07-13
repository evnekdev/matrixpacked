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

Explicit inverses also remain packed:

```rust
use matrixpacked::PackedSPD;

let a = PackedSPD::from_vec(2, vec![4.0_f64, 1.0, 3.0])?;
let inverse = a.inverse()?; // allocates and factorizes packed storage only
assert_eq!(inverse.dimension(), 2);
```

Prefer `solve_vector` (or a reusable factorization's solve methods) when the
goal is applying `A^-1 b`: solving is generally faster and numerically preferable.
Compute an explicit inverse only when the inverse itself is required. The
`inverse_in_place`, `into_inverse`, and `inverse` names respectively mean
overwrite writable packed storage, consume owned packed storage, and allocate
an owned packed result.

Basic packed eigensolvers are available for real symmetric and complex
Hermitian matrices:

```rust
use matrixpacked::PackedSymmetric;

let a = PackedSymmetric::from_vec(2, vec![2.0_f64, 1.0, 2.0])?;
let eigen = a.eigendecomposition()?;
assert_eq!(eigen.eigenvalues, vec![1.0, 3.0]);
```

`eigenvalues()` computes values only. Results are ascending, and eigenvectors
are column-major (`j*n..(j+1)*n`). Borrowed matrices and views clone only their
packed storage because LAPACK overwrites it; `into_eigendecomposition()` reuses
owned packed storage. Hermitian eigensolvers accept only complex Hermitian
matrices, not complex symmetric matrices.

The additive `eigenvalues_divide_conquer()` and
`eigendecomposition_divide_conquer()` methods use `xSPEVD`/`xHPEVD` with
LAPACK workspace queries. Divide-and-conquer often computes eigenvectors faster
for larger matrices, but uses more workspace and may not benefit small matrices.

Selected drivers accept `EigenRange::All`, a zero-based inclusive
`EigenRange::Index { first, last }`, or `EigenRange::Value { lower, upper }`.
Index bounds are checked and converted to LAPACK's one-based indices. Value
selection follows LAPACK exactly: `(lower, upper]`. Selected eigenvectors retain
the same column-major layout, with `count` selected columns of length `dimension`.

Generalized eigensolvers pair a real symmetric or complex Hermitian `A` with a
same-sized `PackedSPD` positive-definite `B`. `GeneralizedEigenproblem` selects
`A x = lambda B x`, `A B x = lambda x`, or `B A x = lambda x`. Basic,
divide-and-conquer, and selected algorithms clone only the two packed operands;
eigenvectors remain column-major. A failed leading principal minor of `B` is
reported as `PositiveDefinitenessFailure` with its one-based index.
Owned operand pairs can use `into_generalized_eigendecomposition` (or its
divide-and-conquer counterpart) to reuse both packed allocations.

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

The `examples/` directory contains individually runnable, assertion-based LAPACK examples covering packed operations for every supported scalar and matrix family.

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
