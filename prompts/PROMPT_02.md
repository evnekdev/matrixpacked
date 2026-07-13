Work on the public GitHub repository:

https://github.com/evnekdev/matrixpacked

Create a fresh branch from the latest `master` after the iterative-refinement PR has been merged.

# Goal

Audit, complete, and standardize packed-matrix inversion APIs across all supported matrix families.

This is primarily an API-quality and correctness task. Do not add new matrix storage types and do not implement inverses by expanding packed matrices into dense matrices.

Use only packed-storage BLAS/LAPACK routines exposed by the Rust `lapack` and `blas` crates.

# Branch

Create:

```text
agent/packed-inverse-api-polish
```

Do not modify `master` directly.

# Families to audit

## Packed triangular

```text
xTPTRI
```

Applicable to:

* `PackedLower`
* `PackedUpper`
* owned storage;
* mutable packed views.

Supported scalars:

* `f32`
* `f64`
* `Complex32`
* `Complex64`

## Positive definite packed

```text
xPPTRI
```

Applicable to Cholesky factorization objects for:

* real SPD;
* complex HPD represented by the crate’s positive-definite packed type.

## Symmetric indefinite packed

```text
xSPTRI
```

Applicable to the factorization produced by `xSPTRF`.

## Hermitian indefinite packed

```text
xHPTRI
```

Applicable to the factorization produced by `xHPTRF`.

# Audit first

Before editing, inventory all existing inversion-related APIs:

```bash
git grep -n "inverse"
git grep -n "tptri\|pptri\|sptri\|hptri"
```

Inspect:

```text
src/backend.rs
src/triangular.rs
src/factorization.rs
src/lower.rs
src/upper.rs
src/spd.rs
src/symmetric.rs
src/hermitian.rs
src/storage.rs
src/error.rs
examples/
```

Produce an internal checklist of:

* routines already implemented;
* scalar families covered;
* owned/view/view-mut support;
* in-place versus allocating APIs;
* naming inconsistencies;
* missing examples;
* missing validation;
* documentation gaps.

Do not rewrite working code merely for style.

# Design requirements

## In-place first

Packed inverse routines overwrite the factorized packed storage.

The core APIs should therefore be in-place:

```rust
factor.inverse_in_place()?;
```

For triangular matrices:

```rust
matrix.inverse_in_place()?;
matrix.inverse_in_place_with_diagonal(Diagonal::Unit)?;
```

## Allocating convenience APIs

Provide allocating wrappers only where they are natural and do not obscure factorization requirements.

Possible pattern:

```rust
let inverse = matrix.inverse()?;
```

For SPD, symmetric-indefinite, and Hermitian-indefinite matrices, this may internally:

1. clone the packed matrix;
2. factorize the clone;
3. invert the factorization in place;
4. return packed inverse storage.

This is acceptable as an explicitly allocating convenience API.

Document that it allocates and destroys the cloned factorization, not the original matrix.

Do not convert to dense storage.

## Factorization-state clarity

For factorization objects, document that after:

```rust
factor.inverse_in_place()?;
```

the factorization storage no longer represents the original factorization; it contains the packed inverse.

If the current type name becomes misleading after mutation, consider returning a packed matrix object rather than continuing to expose it as a factorization.

For example:

```rust
pub fn into_inverse(mut self) -> Result<PackedSPD<T>, PackedMatrixError>
```

This can be safer than mutating a factorization object and leaving its semantic type unchanged.

Evaluate the existing API before selecting the final design.

Prefer a clear ownership transition:

```rust
let inverse = factor.into_inverse()?;
```

while retaining `inverse_in_place()` only where mutable-view workflows require it.

# Storage behavior

Support efficient operation for:

* owned packed matrices;
* mutable packed views;
* factorization objects backed by owned storage;
* factorization objects backed by mutable views, if the current architecture supports them.

Immutable views cannot be inverted in place.

For immutable views, an allocating method may clone into owned packed storage.

Do not allocate a dense matrix.

# Correctness requirements

Verify:

* correct upper/lower orientation passed to LAPACK;
* correct unit/non-unit diagonal behavior for `TPTRI`;
* pivot arrays passed correctly to `SPTRI` and `HPTRI`;
* workspace lengths for real and complex routines;
* Cholesky factorization is required before `PPTRI`;
* failure information is translated through existing error handling;
* singular triangular and indefinite factorizations return errors;
* positive-definite factorization failures are not hidden;
* empty and `1 × 1` matrices behave consistently.

# Reconstruction helpers

An inverse of a symmetric, Hermitian, or SPD matrix remains structurally symmetric or Hermitian.

Ensure the returned packed matrix type preserves that structure.

For example:

* `PackedSPD<T>` may not be the right semantic return type for the inverse unless the type represents both SPD and HPD correctly;
* symmetric-indefinite inverse should return `PackedSymmetric<T>`;
* Hermitian inverse should return `PackedHermitian<T>`;
* triangular inverse should retain lower or upper orientation.

Do not return a generic dense matrix.

# API consistency

Aim for a consistent vocabulary:

```rust
inverse_in_place()
into_inverse()
inverse()
```

Recommended meanings:

* `inverse_in_place`: mutate existing writable packed storage;
* `into_inverse`: consume an owned value or factorization and return an owned packed inverse;
* `inverse`: borrow, allocate packed storage, and return an owned packed inverse.

Avoid alternative names such as:

```text
invert
compute_inverse
make_inverse
```

unless compatibility requires them.

Do not break existing public APIs unnecessarily. If renaming is justified, preserve deprecated forwarding methods where appropriate.

# Examples

Ensure one example exists for each routine and scalar type.

## Triangular

For lower and upper:

```text
lapack_lower_f32_tptri.rs
lapack_lower_f64_tptri.rs
lapack_lower_c32_tptri.rs
lapack_lower_c64_tptri.rs
lapack_upper_f32_tptri.rs
lapack_upper_f64_tptri.rs
lapack_upper_c32_tptri.rs
lapack_upper_c64_tptri.rs
```

## Positive definite

```text
lapack_spd_f32_pptri.rs
lapack_spd_f64_pptri.rs
lapack_spd_c32_pptri.rs
lapack_spd_c64_pptri.rs
```

## Symmetric indefinite

Use only scalar variants supported by the Rust `lapack` crate and the crate’s semantics.

## Hermitian indefinite

```text
lapack_hermitian_c32_hptri.rs
lapack_hermitian_c64_hptri.rs
```

Review existing examples before creating duplicates.

Each example must verify more than successful execution.

Preferred validation:

1. construct `A`;
2. compute packed inverse;
3. multiply `A` by one or more test vectors;
4. multiply the result by `A⁻¹`;
5. verify recovery of the original vectors.

Where multiplication APIs are unavailable, verify packed entries against analytically known inverses for small matrices.

Use floating-point tolerances.

# Tests

Add tests covering:

* lower triangular inverse;
* upper triangular inverse;
* unit diagonal;
* non-unit diagonal;
* singular triangular matrix;
* SPD/HPD inverse;
* symmetric-indefinite inverse;
* Hermitian-indefinite inverse;
* owned storage;
* mutable view;
* empty and `1 × 1` cases;
* real and complex scalars.

Verify the inverse through products, not only hard-coded storage values.

# Documentation

Update:

```text
PACKED_LAPACK_FUNCTIONS.md
EXAMPLE_COVERAGE.md
README.md
```

In the README, add or improve a concise inversion example demonstrating packed storage throughout.

Document that packed inversion is generally more expensive and less numerically preferable than solving systems through a factorization.

Include guidance such as:

* prefer `solve` for applying `A⁻¹b`;
* compute the explicit inverse only when the inverse itself is required.

# Validation

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo check --all-targets
cargo check --examples
cargo test
git diff --check
```

Run all inversion examples with the repository’s supported native backend.

Search for API inconsistencies:

```bash
git grep -n "inverse_in_place\|into_inverse\|pub fn inverse"
git grep -n "todo!\|unimplemented!"
```

Inspect the full diff against `master`.

No unrelated feature work.

# Commit and PR

Commit:

```text
Polish packed inverse APIs
```

Branch:

```text
agent/packed-inverse-api-polish
```

PR title:

```text
Polish packed inverse APIs
```

The PR body must describe:

* APIs audited;
* APIs added or standardized;
* ownership and mutation semantics;
* packed-storage preservation;
* examples and tests;
* any compatibility or deprecation decisions;
* validation results.

Do not stop with only an audit. Complete all reasonable gaps found within this defined inverse scope.

Finish only when the PR is open, no more commits are planned, and the final diff is verified.

End with:

**Safe to rebase and merge.**
