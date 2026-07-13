Work on the public GitHub repository:

https://github.com/evnekdev/matrixpacked

Create a fresh branch from the latest `master`:

```text
agent/nalgebra-test-infrastructure
```

Commit and PR title:

```text
Add nalgebra test infrastructure
```

# Goal

Create a reusable integration-test framework that compares `matrixpacked` packed-storage matrices and operations against independent full-storage calculations using nalgebra.

This PR must establish infrastructure only. Do not attempt to test every numerical operation yet.

# Design principles

1. `nalgebra` must be a dev-dependency only.
2. Production source code must not depend on nalgebra.
3. Oracle calculations must use full matrices and must not reuse the packed-indexing code under test.
4. Test helpers should support:

   * `f32`;
   * `f64`;
   * `Complex32`;
   * `Complex64`.
5. Comparisons must use scale-aware floating-point tolerances.
6. Complex comparisons must compare real and imaginary parts or norms, not rely on exact equality.
7. Test failures must report:

   * matrix dimension;
   * scalar type;
   * index;
   * expected value;
   * actual value;
   * absolute and relative error where useful.

# Dependencies

Add appropriate development dependencies, likely:

```toml
[dev-dependencies]
nalgebra = "..."
approx = "..."
proptest = "..."
rand = "..."
rand_chacha = "..."
```

Before selecting versions, inspect current mutually compatible releases and the crate’s MSRV/toolchain expectations.

Do not add nalgebra to `[dependencies]`.

If `approx` is unnecessary because custom comparison helpers are clearer for generic complex scalars, omit it.

Use deterministic seeded RNGs for repeatable tests.

# Test directory structure

Create a modular integration-test layout similar to:

```text
tests/
    nalgebra_oracle.rs
    oracle/
        mod.rs
        compare.rs
        generate.rs
        convert.rs
        properties.rs
```

Or another valid Rust integration-test structure.

Be aware that every top-level `.rs` file in `tests/` becomes a separate integration-test crate. Organize shared helper modules accordingly.

# Required helper functionality

## 1. Full logical matrix construction

Create helpers that build nalgebra `DMatrix<T>` values independently from packed objects.

For each packed type:

```text
PackedLower
PackedUpper
PackedSymmetric
PackedSPD
PackedHermitian
```

provide test-only conversion helpers such as:

```rust
fn lower_to_dmatrix<T, S>(packed: &PackedLower<T, S>) -> DMatrix<T>
fn upper_to_dmatrix<T, S>(packed: &PackedUpper<T, S>) -> DMatrix<T>
fn symmetric_to_dmatrix<T, S>(packed: &PackedSymmetric<T, S>) -> DMatrix<T>
fn spd_to_dmatrix<T, S>(packed: &PackedSPD<T, S>) -> DMatrix<T>
fn hermitian_to_dmatrix<T, S>(packed: &PackedHermitian<T, S>) -> DMatrix<T>
```

These helpers may use the public logical `(i, j)` accessor, but add at least one separate direct packed-layout decoder that derives indices from documented BLAS packed formulas.

The purpose is to avoid validating the packed layout using exactly the same implementation path as the crate.

## 2. Independent packing helpers

Create test-only functions that pack full matrices into expected traditional BLAS packed layouts.

For example:

```rust
fn pack_lower_column_major<T: Clone>(m: &DMatrix<T>) -> Vec<T>
fn pack_upper_column_major<T: Clone>(m: &DMatrix<T>) -> Vec<T>
```

Use explicit formulas and loops in the test code.

Document the formulas.

These functions must not call `matrixpacked` constructors that derive or rearrange the layout internally.

## 3. Comparison helpers

Create:

```rust
fn assert_scalar_close(...)
fn assert_slice_close(...)
fn assert_matrix_close(...)
fn assert_identity_close(...)
fn assert_hermitian(...)
fn assert_symmetric(...)
fn assert_unitary(...)
fn assert_orthogonal(...)
```

Use a tolerance structure:

```rust
struct Tolerance<R> {
    abs: R,
    rel: R,
}
```

Allow separate defaults for:

```text
f32
f64
Complex32
Complex64
```

Use a comparison resembling:

```text
|actual - expected| <= abs_tol + rel_tol * max(|actual|, |expected|)
```

For matrices, also report a norm-based residual when practical.

## 4. Deterministic matrix generators

Create reusable generators for:

* arbitrary lower triangular matrices;
* arbitrary upper triangular matrices;
* nonsingular triangular matrices;
* unit-diagonal triangular matrices;
* real symmetric matrices;
* complex symmetric matrices;
* Hermitian matrices;
* SPD matrices;
* complex HPD matrices;
* symmetric indefinite matrices;
* Hermitian indefinite matrices;
* vectors;
* column-major multi-RHS buffers.

Use deterministic seeds.

For SPD/HPD matrices, generate:

```text
A = M^T M + shift*I
A = M^H M + shift*I
```

using nalgebra full matrices.

For triangular matrices, force diagonal magnitudes away from zero when tests require invertibility.

For indefinite matrices, construct controlled spectra, such as:

```text
A = Q D Q^T
A = Q D Q^H
```

with both positive and negative diagonal entries.

## 5. Residual helpers

Add helpers for:

```text
||A*x - b||
||A*X - B||
||A*A_inv - I||
||A*v - lambda*v||
||A*v - lambda*B*v||
```

Normalize residuals where useful so tolerances scale with matrix magnitude.

# Initial smoke tests

Add a small set of tests proving the infrastructure itself:

1. lower packing and unpacking round trip;
2. upper packing and unpacking round trip;
3. symmetric logical expansion;
4. Hermitian logical expansion with conjugated off-diagonal entries;
5. SPD generator produces a matrix whose Cholesky decomposition succeeds in nalgebra;
6. matrix comparison detects a deliberately modified entry;
7. deterministic generators reproduce identical matrices for the same seed.

Do not intentionally commit a failing test; test comparison failure behavior indirectly where practical.

# Public API restrictions

Do not add nalgebra conversion methods to the production crate in this PR.

Do not change numerical APIs merely to simplify tests.

If the current public API makes testing impossible, document the issue in the PR rather than expanding production scope silently.

# Validation

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo test
cargo check --all-targets
git diff --check
```

Also run tests more than once to confirm deterministic behavior:

```bash
cargo test
cargo test
```

Inspect:

```bash
git diff master...HEAD --stat
git diff master...HEAD
```

The PR description must explain:

* why nalgebra is a dev-only oracle;
* why independent packing formulas are needed;
* scalar coverage;
* tolerance strategy;
* deterministic random generation;
* test module structure;
* follow-up test modules planned.

Open a draft PR.

Only after all infrastructure tests pass, the final diff is reviewed, and no further commits are planned, finish with:

**Safe to rebase and merge.**
