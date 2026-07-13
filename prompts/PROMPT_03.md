Use this sequence of four separate tasks in:

https://github.com/evnekdev/matrixpacked

Each task must begin from the latest `master` after the preceding PR has been merged.

Never stack the branches unless explicitly instructed.

Do not use handwritten Fortran FFI. Implement only routines exposed by the selected Rust `lapack` crate.

# PR A — Basic packed eigensolvers

Branch:

```text
agent/packed-basic-eigensolvers
```

Goal:

Implement basic packed standard eigenvalue and eigenvector drivers:

```text
xSPEV
xHPEV
```

Supported matrices:

* real packed symmetric: `f32`, `f64`;
* complex packed Hermitian: `Complex32`, `Complex64`.

Do not use complex symmetric matrices for Hermitian eigensolvers.

## Public API

Introduce a clear result type:

```rust
pub struct EigenDecomposition<T, R> {
    pub eigenvalues: Vec<R>,
    pub eigenvectors: Option<Vec<T>>,
    pub dimension: usize,
}
```

Or split results into eigenvalues-only and eigenpairs types if that is cleaner.

Document that eigenvectors are stored column-major, with eigenvector `j` occupying:

```text
vectors[j*n .. (j+1)*n]
```

Suggested methods:

```rust
matrix.eigenvalues()?;
matrix.eigendecomposition()?;
```

or:

```rust
matrix.eigen(Job::ValuesOnly)?;
matrix.eigen(Job::ValuesAndVectors)?;
```

Prefer a typed Rust enum rather than exposing raw LAPACK characters.

For example:

```rust
pub enum Eigenvectors {
    None,
    Compute,
}
```

## Mutation and copying

LAPACK eigensolvers overwrite packed input storage.

For immutable matrices and views:

* clone only the packed storage;
* do not expand to dense storage.

For owned values, optionally provide consuming APIs:

```rust
matrix.into_eigendecomposition()?;
```

This should reuse owned packed storage where possible.

## Ordering

Document that eigenvalues are returned in ascending order, matching LAPACK.

## Examples

Add:

```text
lapack_symmetric_f32_spev.rs
lapack_symmetric_f64_spev.rs
lapack_hermitian_c32_hpev.rs
lapack_hermitian_c64_hpev.rs
```

Each example must verify:

* ascending eigenvalues;
* `A v ≈ λ v`;
* eigenvector normalization;
* eigenvector orthogonality or unitary orthogonality.

Do not validate eigenvectors by exact sign or complex phase.

## Tests

Include:

* eigenvalues only;
* eigenvalues and vectors;
* upper storage;
* lower storage;
* repeated eigenvalues;
* `1 × 1`;
* empty matrix if supported;
* real symmetric;
* complex Hermitian.

## Documentation

Update:

```text
PACKED_LAPACK_FUNCTIONS.md
EXAMPLE_COVERAGE.md
README.md
```

## Validation

Run formatting, checks, tests, examples, and final diff inspection.

Commit:

```text
Implement basic packed eigensolvers
```

PR title:

```text
Implement basic packed eigensolvers
```

Finish with:

**Safe to rebase and merge.**

---

# PR B — Divide-and-conquer packed eigensolvers

Start only after PR A is merged.

Branch:

```text
agent/packed-divide-conquer-eigensolvers
```

Goal:

Implement:

```text
xSPEVD
xHPEVD
```

These are divide-and-conquer eigensolvers.

Use only functions actually exposed by the Rust `lapack` crate.

## API

Reuse the result types and eigenvector-selection enum introduced in PR A.

Do not create a competing API.

Possible methods:

```rust
matrix.eigenvalues_divide_conquer()?;
matrix.eigendecomposition_divide_conquer()?;
```

A better design may introduce an algorithm enum:

```rust
pub enum EigenAlgorithm {
    QR,
    DivideAndConquer,
}
```

However, avoid changing the stable PR A API unnecessarily.

Prefer additive APIs.

## Workspace query

Use LAPACK workspace-query mode where supported:

```text
lwork = -1
lrwork = -1
liwork = -1
```

Verify exact Rust binding signatures and behavior.

Implement checked conversions from returned workspace recommendations to `usize`.

Reject invalid or overflowing recommendations with a crate error rather than panicking.

Do not hard-code oversized workspaces when query mode is available.

## Examples

Add:

```text
lapack_symmetric_f32_spevd.rs
lapack_symmetric_f64_spevd.rs
lapack_hermitian_c32_hpevd.rs
lapack_hermitian_c64_hpevd.rs
```

Validate residuals and orthogonality, not exact eigenvector signs/phases.

## Tests

Test:

* values only;
* values and vectors;
* workspace query handling;
* lower/upper storage;
* repeated eigenvalues;
* real and complex;
* numerical agreement with the basic eigensolver from PR A.

## Documentation

Explain expected tradeoffs:

* divide-and-conquer may use more workspace;
* it is often faster for eigenvectors on larger matrices;
* small matrices may not benefit.

Commit:

```text
Implement divide-and-conquer packed eigensolvers
```

PR title:

```text
Implement divide-and-conquer packed eigensolvers
```

Finish with:

**Safe to rebase and merge.**

---

# PR C — Selected packed eigenvalues and eigenvectors

Start only after PR B is merged.

Branch:

```text
agent/packed-selected-eigensolvers
```

Goal:

Implement:

```text
xSPEVX
xHPEVX
```

These routines select eigenpairs by:

* all eigenvalues;
* index range;
* value interval.

## Typed selection API

Introduce:

```rust
pub enum EigenRange<R> {
    All,
    Index {
        first: usize,
        last: usize,
    },
    Value {
        lower: R,
        upper: R,
    },
}
```

Define clearly whether indices are zero-based and inclusive.

Recommended Rust semantics:

```text
first and last are zero-based and inclusive
```

Convert internally to LAPACK’s one-based indices.

Validate:

* `first <= last`;
* `last < n`;
* finite value bounds where appropriate;
* `lower < upper`, matching LAPACK interval semantics.

Document LAPACK’s value-range convention precisely, typically:

```text
(lower, upper]
```

Do not silently reinterpret it as a fully closed Rust range.

## Result type

Return only the selected eigenvalues and vectors.

Include:

```rust
pub struct SelectedEigenDecomposition<T, R> {
    pub eigenvalues: Vec<R>,
    pub eigenvectors: Option<Vec<T>>,
    pub dimension: usize,
    pub count: usize,
}
```

If LAPACK returns failure indices, preserve them in a useful error or report type rather than discarding them.

## Abstol

Provide a safe default using LAPACK conventions.

Optionally expose an advanced method accepting explicit absolute tolerance.

Avoid making every user understand raw `ABSTOL`.

## Examples

Create examples for:

* real symmetric index range;
* real symmetric value range;
* complex Hermitian index range;
* complex Hermitian value range.

At minimum:

```text
lapack_symmetric_f64_spevx_index.rs
lapack_symmetric_f64_spevx_value.rs
lapack_hermitian_c64_hpevx_index.rs
lapack_hermitian_c64_hpevx_value.rs
```

Add `f32`/`Complex32` examples as needed for scalar coverage.

## Tests

Verify:

* all range;
* index selection;
* value selection;
* invalid ranges;
* no eigenvalues selected;
* eigenvector residuals;
* ascending selected eigenvalues;
* upper/lower packed storage;
* real/complex.

## Documentation

Explain:

* zero-based Rust index API;
* one-based LAPACK conversion;
* value interval convention;
* output vector layout.

Commit:

```text
Implement selected packed eigensolvers
```

PR title:

```text
Implement selected packed eigensolvers
```

Finish with:

**Safe to rebase and merge.**

---

# PR D — Generalized packed eigensolvers

Start only after PR C is merged.

Branch:

```text
agent/packed-generalized-eigensolvers
```

Goal:

Implement generalized packed eigenproblems:

```text
xSPGV
xHPGV
xSPGVD
xHPGVD
xSPGVX
xHPGVX
```

Only implement routines exposed by the Rust `lapack` crate.

Supported problems involve:

* real symmetric `A` and real SPD `B`;
* complex Hermitian `A` and complex HPD `B`.

Do not accept arbitrary symmetric-indefinite `B`.

## Generalized problem enum

Introduce:

```rust
pub enum GeneralizedEigenproblem {
    AxEqualsLambdaBx,
    ABxEqualsLambdaX,
    BAxEqualsLambdaX,
}
```

Map these to LAPACK `ITYPE = 1, 2, 3`.

Document the mathematical meaning of each variant.

## Type-safe operands

Design signatures that require compatible dimensions and scalar types.

Example:

```rust
a.generalized_eigendecomposition(&b, problem)?;
```

Validate:

* equal dimensions;
* compatible packed orientation where required;
* valid SPD/HPD `B`;
* storage lengths.

LAPACK overwrites both `A` and `B`.

For borrowed inputs, clone only packed storage.

Provide consuming APIs where useful to avoid duplicate copying:

```rust
a.into_generalized_eigendecomposition(b, problem)?;
```

## Algorithms

Support in logical stages:

1. basic driver:

   * `SPGV`
   * `HPGV`
2. divide-and-conquer:

   * `SPGVD`
   * `HPGVD`
3. selected eigenpairs:

   * `SPGVX`
   * `HPGVX`

Reuse:

* result types;
* eigenvector-selection enum;
* eigen-range enum;
* workspace helpers;

from the standard eigensolver PRs.

Do not duplicate nearly identical public types.

## Errors

Distinguish:

* illegal LAPACK argument;
* failure to converge;
* `B` not positive definite;
* invalid range;
* dimension mismatch.

When LAPACK reports that a leading principal minor of `B` is not positive definite, map this to a meaningful crate error with the failing index.

## Examples

Create at least:

```text
lapack_symmetric_f64_spgv.rs
lapack_symmetric_f64_spgvd.rs
lapack_symmetric_f64_spgvx.rs
lapack_hermitian_c64_hpgv.rs
lapack_hermitian_c64_hpgvd.rs
lapack_hermitian_c64_hpgvx.rs
```

Add `f32` and `Complex32` examples so all supported scalar backends are compiled and exercised.

Validate generalized residuals according to `ITYPE`.

For the common type-1 problem verify:

```text
A v ≈ λ B v
```

Do not validate generalized eigenvectors by exact signs or phases.

## Tests

Cover:

* all three generalized problem types;
* values only;
* values and vectors;
* basic and divide-and-conquer algorithms;
* selected index/value ranges;
* invalid dimensions;
* non-positive-definite `B`;
* upper/lower storage;
* real symmetric;
* complex Hermitian.

## Documentation

Update the status table and README with:

* mathematical definitions;
* SPD/HPD requirement for `B`;
* eigenvector layout;
* mutation/allocation behavior;
* algorithm choices.

## Validation

Run all formatting, checks, tests, and examples.

Inspect the full diff and ensure no dense matrix fallback was introduced.

Commit:

```text
Implement generalized packed eigensolvers
```

PR title:

```text
Implement generalized packed eigensolvers
```

Finish only after the PR is complete and open.

End with:

**Safe to rebase and merge.**
