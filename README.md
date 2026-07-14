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

For a single solve, `solve_once(&rhs, nrhs)` calls LAPACK's combined packed
factor-and-solve driver while leaving both inputs unchanged. Owned matrices can
use `into_solve_once`, and writable packed views can use
`solve_once_in_place` to avoid copying packed storage. Right-hand sides are
column-major. When solving repeatedly with the same matrix, retain a Cholesky,
symmetric, or Hermitian factorization instead of refactorizing each time.

Those reusable factors also provide determinant diagnostics without expanding
the matrix. Cholesky factors expose `log_determinant()` and `determinant()`;
real symmetric and complex Hermitian Bunch-Kaufman factors expose `slogdet()`,
`inertia()`, and an explicit-tolerance `inertia_with_tolerance()`. Prefer the
logarithmic form because a direct determinant product can overflow or
underflow. Inertia comes directly from the factor's 1-by-1 and 2-by-2 diagonal
blocks and therefore does not require an eigenvalue decomposition. Exact
`inertia()` uses exact floating-point zero; the tolerance-aware method never
applies a hidden threshold. Complex-symmetric factors deliberately do not use
the Hermitian real-sign diagnostics because their determinants can have phase.

Packed lower and upper triangular matrices expose `determinant`,
`log_abs_determinant`, and exact `is_singular` methods. Pass `Diagonal::Unit`
when the stored diagonal is to be treated as one.

Use `expert_solve` when a one-shot solve also needs a reciprocal condition
estimate and forward/backward error bounds. Positive-definite systems can use
`expert_solve_with_options` and `EquilibrationMode::Compute` for LAPACK-managed
equilibration. The current expert API computes its own factorization.

Real symmetric and complex Hermitian matrices support in-place BLAS packed
rank-1 and rank-2 updates, including explicitly strided vectors. These methods
require mutable packed storage and do not clone the matrix. Because an
unrestricted negative update can destroy positive definiteness, `PackedSPD`
does not expose them directly: call `into_symmetric()` for real matrices or
`into_hermitian()` for complex matrices to intentionally obtain the less
restrictive structure first.

Positive-definite packed matrices can compute LAPACK equilibration diagnostics
without modifying their storage. `equilibration()` returns positive scaling
factors, their minimum-to-maximum ratio, and the largest diagonal entry.
Writable matrices can use `equilibrate_in_place()` or
`apply_equilibration_in_place()`; both scale entries directly in packed storage
as `s[i] * A(i,j) * s[j]`, without expanding to a dense matrix. Complex HPD
diagonals follow LAPACK's real-diagonal convention.

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

For workflows that need the intermediate standard problem,
`generalized_reduction` accepts a previously computed packed Cholesky factor
`B = L L^H`. With the crate's lower traditional-packed storage it produces
`L^-1 A L^-H` for `A x = lambda B x`, and `L^H A L` for both
`A B x = lambda x` and `B A x = lambda x`. After solving the standard problem,
recover original eigenvectors with `x = L^-H y` for the first two forms and
`x = L y` for the third. The borrowing method clones only packed `A`,
`reduce_generalized_in_place` overwrites mutable storage without allocation,
and `into_generalized_reduction` reuses owned storage; all borrow the unchanged
Cholesky factor. (`H` is transpose for real matrices.)

Low-level `tridiagonal_reduction` APIs retain LAPACK's packed reflector
representation and expose the real tridiagonal diagonal/off-diagonal arrays.
Use `generate_q` for a column-major dense orthogonal/unitary matrix, or apply
the reflectors directly to a column-major dense target with
`apply_q_in_place`. These APIs are intended for workflows that need the
intermediate tridiagonal form; the high-level eigensolvers remain simpler.

Triangular format conversions distinguish three representations explicitly:
`PackedLower`/`PackedUpper` use traditional packed (`TP`) storage,
`FullTriangular` uses an `n x n` column-major full (`TR`) buffer, and
`RectangularFullPacked` uses compact rectangular full packed (`TF`/RFP)
storage. Use `to_full_triangular` and `from_full_triangular` for `TP <-> TR`,
or `to_rectangular_full_packed` and `from_rectangular_full_packed` for
`TP <-> RFP`. `RfpTranspose` records normal versus transposed physical layout;
for complex scalars the latter is LAPACK's conjugate-transposed RFP form. RFP
retains `n*(n+1)/2` values but is a distinct representation, so convert it back
before calling ordinary packed-matrix methods.

Nalgebra interoperability is optional behind the `nalgebra-interop` feature.
With it enabled, `FullTriangular::to_dmatrix` clones the full `n x n`
column-major buffer into a `nalgebra::DMatrix`, while `into_dmatrix` moves and
reuses the owned buffer. These are owned full-storage conversions, not
zero-copy views. `FullTriangular::try_from_dmatrix` accepts a selected triangle,
requires a square matrix, copies nalgebra's compatible column-major storage,
and zeros the opposite triangle to preserve `FullTriangular`'s structural-zero
invariant.

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

## Testing

The repository separates fast deterministic tests, reproducible proptests,
and opt-in extended numerical stress tests. See [`TESTING.md`](TESTING.md) for
backend-specific commands, CI case counts, failure reproduction, conditioning
categories, expected runtime, and oracle limitations.
