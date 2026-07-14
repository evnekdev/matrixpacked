# matrixpacked

`matrixpacked` provides owned and borrowed square matrices in traditional
BLAS/LAPACK packed-column storage. It is for applications that already work
with packed triangular, symmetric, Hermitian, or positive-definite data and
want packed kernels without first expanding to a dense matrix.

Start by choosing a [matrix family](#matrix-family-map), understanding the
[physical layout](#exact-packed-layout), selecting an
[ownership form](#ownership-and-allocation), and arranging a
[native provider](#native-blaslapack-providers) before linking numerical code.

## Purpose and design

An order-`n` packed matrix stores `n(n+1)/2` scalars instead of `n²`. Methods
call traditional packed BLAS/LAPACK families such as `xTPMV`, `xPPTRF`,
`xSPTRF`, and `xHPEV` directly. This avoids an `O(n²)` dense expansion for
operations that have a packed kernel and keeps the caller's packed layout
visible throughout the API.

Packed storage is not universally faster. Dense Level-3 BLAS algorithms use
cache and parallel hardware more effectively, and many libraries focus their
optimization effort there. Packed formats are most useful when memory, wire
formats, an existing LAPACK interface, or compatibility with stored data
matters more than peak throughput on large matrices.

## Matrix family map

| Family | Structure | Physical data | Logical unstored entries |
|---|---|---|---|
| [`PackedLower`] | lower triangular | lower columns | zero above diagonal |
| [`PackedUpper`] | upper triangular | upper columns | zero below diagonal |
| [`PackedSymmetric`] | `Aᵀ = A` | lower columns | mirror, no conjugation |
| [`PackedHermitian`] | `Aᴴ = A` | lower columns | conjugate mirror |
| [`PackedSPD`] | SPD/HPD intent | lower columns | symmetric/Hermitian mirror |

The LAPACK scalar set is `f32`, `f64`, [`num_complex::Complex32`], and
[`num_complex::Complex64`] where the underlying routine exists. Some operations
are narrower: ordinary symmetric BLAS rank updates are real, while Hermitian
operations are complex.

Three distinctions are especially important:

- `PackedSymmetric<Complex<_>>` means `Aᵀ = A`; it does **not** conjugate.
- `PackedHermitian<Complex<_>>` means `Aᴴ = A` and conjugates mirrored reads.
- `PackedSPD<Complex<_>>` records HPD intent. Basic constructors validate the
  packed length, not positive definiteness. Cholesky-backed operations perform
  the numerical check.

## Exact packed layout

Coordinates are zero-based. Lower packed-column storage for `n = 3` is:

```text
physical: [a00, a10, a20, a11, a21, a22]

logical:  a00   .    .
          a10  a11   .
          a20  a21  a22
```

For `row >= column`, the physical index is:

```text
column * (2*n - column + 1) / 2 + (row - column)
```

Upper packed-column storage for `n = 3` is:

```text
physical: [a00, a01, a11, a02, a12, a22]

logical:  a00  a01  a02
           .   a11  a12
           .    .   a22
```

For `row <= column`, the physical index is:

```text
column * (column + 1) / 2 + row
```

Triangular `get` returns zero for a logical structural zero. Symmetric,
Hermitian, and SPD/HPD access maps either triangle to stored lower data, with
conjugation for complex Hermitian behavior. Stored-only mutable access rejects
structural zeros; indexing panics on invalid or unstored coordinates.

```rust
use matrixpacked::PackedLower;

let mut a = PackedLower::from_vec(
    3,
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
assert_eq!(a.get(0, 2)?, 0.0);
assert_eq!(a[(2, 1)], 5.0);
a[(2, 1)] = 9.0;
assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0, 4.0, 9.0, 6.0]);
# Ok::<(), matrixpacked::PackedMatrixError>(())
```

## Ownership and allocation

The default storage parameter is `Vec<T>`, so `PackedLower<T>` is owned.
Aliases ending in `View` borrow `&[T]`; aliases ending in `ViewMut` borrow
`&mut [T]`. Views are zero-copy and numerical methods operate directly on the
borrowed packed buffer when their mutability permits.

```rust
use matrixpacked::PackedSymmetric;

let mut storage = [2.0_f64, 1.0, 3.0];
{
    let mut view = PackedSymmetric::from_slice_mut(2, &mut storage)?;
    view.set(0, 1, 4.0)?;
}
assert_eq!(storage, [2.0, 4.0, 3.0]);
# Ok::<(), matrixpacked::PackedMatrixError>(())
```

Method names communicate ownership:

- borrowing methods leave inputs unchanged and copy packed storage when LAPACK
  must overwrite it;
- `_in_place` methods overwrite caller-owned storage;
- `into_` methods consume owned storage and reuse its allocation;
- methods returning a dense or nalgebra matrix allocate `n²` scalars.

Factorizations and reductions are destructive at the LAPACK level. Keep the
original matrix when refinement or an original-matrix norm is needed. After a
factor's `inverse_in_place`, its buffer contains the inverse and is no longer a
valid factorization.

## Basic operations

Construct owned data with `from_vec`, generate stored coordinates with
`from_fn`, or borrow slices with `from_slice` and `from_slice_mut`. `get` and
`set` provide checked logical access; `as_slice` exposes physical packed order.

Elementwise arithmetic operates on corresponding stored values. Binary
operators panic on dimension mismatch; `component_mul` and `component_div`
return [`PackedMatrixError`] instead. Matrix-vector products use packed BLAS.
Triangular methods expose transpose, conjugate-transpose, and unit-diagonal
choices through [`Transpose`] and [`Diagonal`].

Real symmetric and complex Hermitian matrices support packed rank-1 and rank-2
updates. Strided variants follow BLAS increment rules. An unrestricted negative
update can destroy positive definiteness, so [`PackedSPD`] requires conversion
to [`PackedSymmetric`] or [`PackedHermitian`] before those updates are available.

## Linear systems

There are four solve levels:

1. `solve_vector` is concise and factors a temporary copy.
2. `solve_once` accepts one or more column-major right-hand sides and uses a
   combined factor-and-solve driver.
3. A reusable [`PackedCholesky`], [`PackedSymmetricFactor`], or
   [`PackedHermitianFactor`] amortizes factorization across repeated solves.
4. `expert_solve` adds reciprocal condition and forward/backward error
   estimates; SPD/HPD expert solves can request equilibration.

For `nrhs` right-hand sides, buffers contain consecutive length-`n` columns.
The reusable-factor path is normally the right choice for repeated solves.

```no_run
use matrixpacked::{MatrixNorm, PackedSPD};

let a = PackedSPD::from_vec(2, vec![4.0_f64, 1.0, 3.0])?;
let factor = a.cholesky()?;
let mut x = factor.solve_vector(&[6.0, 7.0])?;
let report = factor.refine_vector_in_place(&a, &[6.0, 7.0], &mut x)?;
let rcond = factor.rcond(a.matrix_norm(MatrixNorm::One)?)?;
assert_eq!(report.forward_error.len(), 1);
assert!(rcond > 0.0);
# Ok::<(), matrixpacked::PackedMatrixError>(())
```

Refinement requires both the original matrix and its matching factor. It
updates an approximate solution and returns one forward estimate (`ferr`) and
one componentwise backward estimate (`berr`) per RHS.

## Inverse, norms, conditions, and diagnostics

Prefer solving `A x = b` over forming `A⁻¹`: it does less work, uses less
storage, and is usually numerically preferable. Explicit packed inverses are
available when the inverse itself is required.

[`MatrixNorm`] selects max-absolute, one, infinity, or Frobenius norms where
supported. Reciprocal condition methods estimate `1 / cond(A)`; a value near
zero indicates sensitivity or singularity. Estimators are approximate and may
vary slightly by provider.

Cholesky factors expose determinant and log-determinant diagnostics. Pivoted
symmetric/Hermitian factors expose [`SignedLogDet`], [`Inertia`], and
singularity tests derived from their 1-by-1 and 2-by-2 blocks. The logarithmic
form avoids overflow and underflow. Tolerance-aware inertia uses the caller's
explicit threshold and never introduces a hidden one.

## Eigenproblems and reductions

Real symmetric and complex Hermitian matrices provide:

- basic eigenvalue/eigenvector drivers;
- divide-and-conquer drivers, often faster for larger eigenvector problems but
  with more workspace;
- selected drivers using all values, a zero-based inclusive index range, or
  LAPACK's half-open value interval `(lower, upper]`;
- generalized symmetric/Hermitian-definite drivers with [`PackedSPD`] `B`.

Eigenvalues are ascending. Eigenvectors are consecutive column-major columns.
A real eigenvector is defined only up to sign, a complex vector up to unit
phase, and a repeated eigenspace may use any valid orthonormal basis.

[`GeneralizedEigenproblem`] selects `A x = λ B x`, `A B x = λ x`, or
`B A x = λ x` and documents the associated normalization. Lower-level
generalized reduction accepts a Cholesky factor of `B`. Tridiagonal reduction
retains Householder reflectors, exposes the real diagonal/off-diagonal arrays,
and can generate dense `Q` or apply reflectors to a dense column-major target.

## Format conversion and nalgebra interoperability

LAPACK format conversion distinguishes:

- TP: traditional [`PackedLower`] or [`PackedUpper`] storage;
- TR: [`FullTriangular`], an `n × n` column-major triangular buffer;
- TF/RFP: [`RectangularFullPacked`], a compact rectangular rearrangement.

TP ↔ TR and TP ↔ RFP use LAPACK conversion routines. RFP stays compact but is
a distinct physical layout and must be converted back before ordinary packed
operations.

The optional `nalgebra-interop` feature adds owned `nalgebra::DMatrix`
conversions. It provides conversions, not views: nalgebra's rectangular stride
model cannot describe variable-length packed columns. Structured expansion,
tolerance validation, and nalgebra Cholesky-backed SPD/HPD validation are pure
Rust. Triangular nalgebra conversion currently follows the TP ↔ TR LAPACK path
and therefore needs a native provider at final link.

Extraction constructors ignore the opposite triangle. Strict
`try_from_dmatrix` constructors validate it with:

```text
|a - b| <= absolute + relative * max(|a|, |b|)
```

Strict SPD/HPD conversion validates structure and then proves positive
definiteness with nalgebra Cholesky. See the
[nalgebra guide](https://github.com/evnekdev/matrixpacked/blob/master/NALGEBRA_INTEROP.md)
for the complete conversion matrix and allocation costs.

## Native BLAS/LAPACK providers

The Rust `blas` and `lapack` crates are bindings; a numerical executable must
link one native implementation. `matrixpacked` selects no provider by default
so it can coexist with an application's existing BLAS/LAPACK choice.

- On Linux, `openblas-static` bundles OpenBLAS.
- On Windows, use `intel-mkl-static` for the repository's supported bundled
  provider workflow. Do not assume the non-vcpkg OpenBLAS source build works on
  Windows.
- Applications may arrange compatible native symbols themselves.

```toml
[dependencies]
matrixpacked = { version = "0.1", features = ["openblas-static"] }
```

Select only one bundled provider feature. docs.rs enables
`nalgebra-interop` for API visibility but does not link and execute numerical
routines while generating ordinary Rustdoc.

## Testing and reproducibility

The repository separates deterministic unit/integration tests, fixed-seed
nalgebra oracle properties, and an opt-in extended tier for larger and
ill-conditioned problems. Linux uses OpenBLAS-linked commands; Windows uses
MKL-linked commands. Property counts and seeds are reproducible with
`MATRIXPACKED_PROPTEST_CASES` and `MATRIXPACKED_PROPTEST_SEED`. Extended tests
also require `MATRIXPACKED_EXTENDED_TESTS=1`.

```text
# Linux
cargo test --all-targets --features openblas-static

# Windows
cargo test --all-targets --features intel-mkl-static
```

See the [testing guide](https://github.com/evnekdev/matrixpacked/blob/master/TESTING.md)
for backend-qualified commands, replay instructions, runtime, and oracle
limitations. The
[example matrix](https://github.com/evnekdev/matrixpacked/blob/master/EXAMPLE_COVERAGE.md)
and [routine status](https://github.com/evnekdev/matrixpacked/blob/master/PACKED_LAPACK_FUNCTIONS.md)
remain the detailed sources of truth.

## Errors and numerical behavior

[`PackedMatrixError`] separates layout and dimension mistakes from LAPACK
argument failures and numerical failures. Common outcomes include a singular
triangular diagonal or pivot block, failure of an SPD/HPD leading minor,
eigensolver non-convergence, and invalid selected ranges.

Floating-point results are approximate. Compare residuals or use
scale-sensitive tolerances rather than exact equality. Ill-conditioned systems
can have a small residual but a sensitive solution, so inspect reciprocal
condition and refinement estimates. Eigenvectors require sign/phase- and
subspace-aware comparison.

## Features and support

| Feature | Effect | Provider selected? |
|---|---|---|
| default | core packed API | no |
| `nalgebra-interop` | owned `DMatrix` conversion/validation | no |
| `openblas-static` | static OpenBLAS | yes, Linux workflow |
| `intel-mkl-static` | sequential LP64 MKL | yes, Windows workflow |

| Scalar | Triangular | Symmetric | Hermitian | SPD/HPD |
|---|---|---|---|---|
| `f32`, `f64` | yes | yes | not distinct | yes |
| `Complex32`, `Complex64` | yes | where supported | yes | HPD |

Exact routine/scalar coverage evolves. Consult
[PACKED_LAPACK_FUNCTIONS.md](https://github.com/evnekdev/matrixpacked/blob/master/PACKED_LAPACK_FUNCTIONS.md)
instead of inferring availability from the summary table.
