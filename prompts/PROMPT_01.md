Work on the public GitHub repository:

https://github.com/evnekdev/matrixpacked

Create a new branch from the latest `master`.

This task must start only after the packed condition-estimation PR has been merged.

# Goal

Implement complete packed-storage iterative refinement and error estimation using only routines exposed by the Rust `lapack` crate.

Do not add handwritten `extern "C"` declarations or direct Fortran-symbol bindings.

Implement:

* `xTPRFS` for packed triangular matrices, reviewing the existing implementation
* `xPPRFS` for positive-definite packed matrices
* `xSPRFS` for symmetric-indefinite packed matrices
* `xHPRFS` for Hermitian-indefinite packed matrices

Supported scalar families:

| Family  | Scalars                                                                                   |
| ------- | ----------------------------------------------------------------------------------------- |
| `TPRFS` | `f32`, `f64`, `Complex32`, `Complex64`                                                    |
| `PPRFS` | `f32`, `f64`, `Complex32`, `Complex64`                                                    |
| `SPRFS` | `f32`, `f64` unless the Rust `lapack` crate explicitly exposes complex symmetric variants |
| `HPRFS` | `Complex32`, `Complex64`                                                                  |

Verify the exact functions exposed by the installed Rust `lapack` crate before coding. Do not invent bindings for unavailable routines.

# Repository workflow

1. Fetch the latest remote state.
2. Switch to `master`.
3. Pull using fast-forward only.
4. Create a fresh branch:

```text
agent/packed-iterative-refinement
```

5. Do not modify `master` directly.
6. Keep the PR limited to iterative refinement, examples, tests, and related documentation.
7. Do not continue committing after declaring the branch safe to merge.

# Existing API review

Inspect:

```text
src/backend.rs
src/triangular.rs
src/factorization.rs
src/spd.rs
src/symmetric.rs
src/hermitian.rs
src/lib.rs
src/error.rs
```

Find the existing:

```rust
RefinementReport<R>
```

and the triangular method resembling:

```rust
refine_many_in_place(...)
```

Review the existing triangular implementation for:

* correct LAPACK workspace sizes;
* correct real versus complex workspace handling;
* correct `ldb` and `ldx`;
* correct right-hand-side layout;
* zero-size handling;
* no unnecessary matrix copies;
* correct error propagation.

Preserve compatible public APIs where practical.

# Public result type

Reuse or improve the existing result type:

```rust
pub struct RefinementReport<R> {
    pub forward_error: Vec<R>,
    pub backward_error: Vec<R>,
}
```

Document clearly:

* one `forward_error` value per right-hand side;
* one `backward_error` value per right-hand side;
* the solution matrix is modified in place;
* right-hand sides use column-major `n × nrhs` storage.

Do not include a duplicate solution vector in the report when the API already modifies `x` in place.

# API design

Expose iterative refinement from the object that naturally owns the required factorization.

Preferred pattern:

```rust
let factor = a.cholesky()?;

let mut x = initial_solution;
let report = factor.refine_many_in_place(
    a.as_slice(),
    &b,
    &mut x,
    nrhs,
)?;
```

However, inspect the current factorization types before choosing signatures.

The refinement routines require both:

* the original matrix;
* its factorization.

Design the API so this requirement is explicit and type-safe.

Possible approaches:

```rust
factor.refine_many_in_place(&original, &b, &mut x, nrhs)
```

or:

```rust
original.refine_many_with_factor_in_place(&factor, &b, &mut x, nrhs)
```

Prefer the approach most consistent with the current crate.

Do not silently refactorize.

# Required methods

Provide an in-place multi-right-hand-side method for each relevant family.

Examples of intended semantics:

```rust
pub fn refine_many_in_place(
    &self,
    original: &PackedSPDView<T>,
    b: &[T],
    x: &mut [T],
    nrhs: usize,
) -> Result<RefinementReport<T::Real>, PackedMatrixError>
```

Also provide convenient one-vector wrappers where consistent:

```rust
pub fn refine_vector_in_place(
    ...
) -> Result<RefinementReport<T::Real>, PackedMatrixError>
```

Avoid excessive wrapper proliferation. The minimum useful API is:

* multi-RHS in-place refinement;
* optionally a one-RHS convenience wrapper.

# Backend implementation

Extend the internal backend traits only with functions available from the Rust `lapack` crate.

Implement dispatch separately for:

* `f32`
* `f64`
* `Complex32`
* `Complex64`

Use:

```rust
lapack::spprfs
lapack::dpprfs
lapack::cpprfs
lapack::zpprfs
```

and corresponding available `SPRFS`/`HPRFS` routines.

Verify exact Rust signatures from the dependency source. Do not assume argument lists from Netlib documentation alone.

# Workspace requirements

Allocate only the workspace required by LAPACK.

Carefully distinguish real and complex routines.

Typical patterns may involve:

* real work arrays;
* complex work arrays;
* integer work arrays;
* `ferr`;
* `berr`.

Do not use one generic workspace formula unless it is valid for all scalar families.

Add internal helper functions where they reduce duplication without obscuring LAPACK requirements.

# Validation and input checks

Validate before calling LAPACK:

* original matrix and factorization dimensions match;
* packed storage lengths are correct;
* `b.len() == n * nrhs`;
* `x.len() == n * nrhs`;
* `n`, `nrhs`, `ldb`, and `ldx` fit in `i32`;
* factorization orientation matches the original matrix;
* zero-dimensional matrices are handled consistently.

Return existing `PackedMatrixError` variants where possible.

Add new error variants only when existing ones cannot describe the failure clearly.

No panics for user-provided dimensions or slice lengths.

# Examples

Add focused examples for every supported scalar and family.

## Positive definite

```text
examples/lapack_spd_f32_pprfs.rs
examples/lapack_spd_f64_pprfs.rs
examples/lapack_spd_c32_pprfs.rs
examples/lapack_spd_c64_pprfs.rs
```

## Symmetric indefinite

At minimum:

```text
examples/lapack_symmetric_f32_sprfs.rs
examples/lapack_symmetric_f64_sprfs.rs
```

Add complex symmetric examples only if the Rust `lapack` crate exposes corresponding routines and the crate already supports those factorization semantics.

## Hermitian indefinite

```text
examples/lapack_hermitian_c32_hprfs.rs
examples/lapack_hermitian_c64_hprfs.rs
```

Review and retain the existing triangular `TPRFS` examples.

Each example must:

1. construct a small packed matrix;
2. factorize it;
3. construct an exact or deliberately perturbed initial solution;
4. call refinement;
5. verify that the refined solution is close to the expected solution;
6. verify that `forward_error` and `backward_error` contain one value per RHS;
7. avoid brittle assertions that error estimates must equal exact constants.

Use tolerance-based assertions.

# Tests

Add unit or integration tests for:

* one RHS;
* multiple RHS;
* invalid RHS length;
* invalid solution length;
* dimension mismatch between original matrix and factorization;
* real scalar;
* complex scalar;
* upper and lower packed storage where both are supported.

Tests should verify both correctness and API validation.

# Documentation

Update:

```text
PACKED_LAPACK_FUNCTIONS.md
EXAMPLE_COVERAGE.md
```

Mark implemented routines accurately:

```text
TPRFS
PPRFS
SPRFS
HPRFS
```

Do not mark unavailable scalar variants as implemented.

Document the distinction between:

* factorization;
* solve;
* refinement;
* forward error estimate;
* backward error estimate.

Update the checked-example count from the actual repository contents.

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

Run the LAPACK example runner with the repository’s normal backend feature if the environment supports native linking.

Also run repository searches:

```bash
git grep -n "PPRFS\|SPRFS\|HPRFS\|TPRFS"
git grep -n "todo!\|unimplemented!"
```

Inspect:

```bash
git diff master...HEAD --stat
git diff master...HEAD
```

Confirm there are no unrelated changes.

# Commit and PR

Use one logical commit:

```text
Implement packed iterative refinement
```

Push:

```text
agent/packed-iterative-refinement
```

Open a draft PR targeting `master`.

Suggested title:

```text
Implement packed iterative refinement
```

The PR description must include:

* implemented routine families;
* supported scalar types;
* public API summary;
* workspace and allocation approach;
* examples and tests added;
* validation commands and results;
* any LAPACK routines intentionally omitted because the Rust binding crate does not expose them.

Do not declare the task complete while additional commits are planned.

Only after the final diff is verified, checks pass as far as the environment permits, and the PR is open, finish with this exact standalone line:

**Safe to rebase and merge.**
