# Prompt 01 — Add optional nalgebra interoperability and `FullTriangular` conversions

Work on the public GitHub repository:

https://github.com/evnekdev/matrixpacked

## Starting conditions

This is the first task in the nalgebra interoperability series.

1. Fetch all remote changes.
2. Switch to `master`.
3. Pull using fast-forward only.
4. Confirm the working tree is clean.
5. Create a fresh branch:

```text
agent/nalgebra-interop-foundation
```

6. Do not modify `master` directly.
7. Do not stack this branch on another unmerged feature branch.
8. Keep the PR limited to:
   - optional nalgebra dependency configuration;
   - conversions between `FullTriangular<T>` and nalgebra matrices;
   - focused tests and documentation.

## Goal

Introduce optional nalgebra interoperability without making nalgebra a mandatory runtime dependency.

Add conversions between:

```text
matrixpacked::FullTriangular<T>
nalgebra::DMatrix<T>
```

This first PR must establish the feature boundary and the simplest full-storage conversion before adding packed matrix conversions.

## Dependency configuration

Nalgebra is currently used as a development dependency for oracle tests.

Add nalgebra as an optional normal dependency while retaining its use in tests.

Use a feature named:

```text
nalgebra-interop
```

Preferred Cargo configuration:

```toml
[features]
default = []
nalgebra-interop = ["dep:nalgebra"]
```

Add an optional dependency similar to:

```toml
[dependencies.nalgebra]
version = "0.35"
optional = true
default-features = false
features = ["std"]
```

Before editing, inspect the current `Cargo.toml`.

Do not duplicate nalgebra with incompatible versions in `[dependencies]` and `[dev-dependencies]`.

Use one compatible version specification.

A valid arrangement may be:

```toml
[dependencies.nalgebra]
version = "0.35"
optional = true
default-features = false
features = ["std"]

[dev-dependencies]
nalgebra = { version = "0.35", default-features = false, features = ["std"] }
```

Cargo can unify these, but verify the dependency graph with:

```bash
cargo tree -i nalgebra
```

Do not add nalgebra’s BLAS or LAPACK integration features.

## Module layout

Create a feature-gated interoperability module, preferably:

```text
src/nalgebra_interop.rs
```

In `src/lib.rs`:

```rust
#[cfg(feature = "nalgebra-interop")]
mod nalgebra_interop;
```

Keep the implementation internal if methods and trait implementations are attached to existing public types.

Expose a public module only if it contains public interoperability-specific types or traits.

Do not expose implementation helpers unnecessarily.

## `FullTriangular<T>` to `DMatrix<T>`

`FullTriangular<T>` already stores exactly `n * n` values in column-major order.

Implement:

```rust
#[cfg(feature = "nalgebra-interop")]
impl<T> FullTriangular<T>
where
    T: nalgebra::Scalar,
{
    pub fn to_dmatrix(&self) -> nalgebra::DMatrix<T>
    where
        T: Clone;

    pub fn into_dmatrix(self) -> nalgebra::DMatrix<T>;
}
```

Adapt bounds to the actual nalgebra API.

### Requirements

- `to_dmatrix()` clones the full column-major buffer.
- `into_dmatrix()` reuses the owned `Vec<T>` without cloning if nalgebra permits constructing a matrix from the vector.
- Preserve the exact logical shape `n × n`.
- Preserve column-major ordering.
- Do not call LAPACK for this conversion.
- Do not reconstruct entries through logical indexing.
- Do not allocate more than the resulting full matrix requires.

Use an appropriate constructor such as:

```rust
DMatrix::from_vec(n, n, data)
```

Verify nalgebra’s storage order before relying on it.

## `DMatrix<T>` to `FullTriangular<T>`

Implement fallible constructors:

```rust
#[cfg(feature = "nalgebra-interop")]
impl<T> FullTriangular<T>
where
    T: nalgebra::Scalar + Clone,
{
    pub fn try_from_dmatrix(
        matrix: &nalgebra::DMatrix<T>,
        triangle: Triangle,
    ) -> Result<Self, PackedMatrixError>;

    pub fn from_dmatrix_triangle(
        matrix: &nalgebra::DMatrix<T>,
        triangle: Triangle,
    ) -> Result<Self, PackedMatrixError>;
}
```

Select one clear public name rather than implementing duplicate aliases.

### Semantics

A `FullTriangular<T>` contains an `n × n` full buffer, but only the selected triangle is semantically relevant.

The conversion must:

1. require a square matrix;
2. copy its full column-major storage;
3. record the selected `Triangle`;
4. preserve values in both halves unless the current `FullTriangular` invariant requires structural zeros outside the selected triangle.

Inspect the current contract carefully.

If `FullTriangular` promises structural zeros outside the selected triangle, explicitly zero the opposite triangle using `T::zero()`.

If it merely states that the opposite triangle is ignored, preserve the full data and document that only the selected triangle participates in conversion back to packed storage.

Do not silently change the existing invariant.

## Standard conversion traits

Consider implementing:

```rust
impl<T> From<FullTriangular<T>> for DMatrix<T>
```

only if the trait bounds remain clear and the conversion is infallible.

Consider:

```rust
impl<T> TryFrom<&DMatrix<T>> for FullTriangular<T>
```

but note that the target `Triangle` cannot be inferred.

Therefore, an inherent constructor taking `Triangle` is likely more appropriate.

Do not introduce a misleading `TryFrom` implementation that arbitrarily assumes lower or upper storage.

## Error handling

If a matrix is not square, return a meaningful crate error.

Inspect `PackedMatrixError`.

Reuse an existing shape or dimension error if available.

If no suitable variant exists, add a narrowly scoped variant such as:

```rust
NonSquareMatrix {
    rows: usize,
    columns: usize,
}
```

Document it.

Do not panic on non-square input.

## Tests

Add feature-gated integration tests.

Suggested file:

```text
tests/nalgebra_full_triangular.rs
```

Ensure the entire test is gated correctly when the feature is disabled.

Test:

1. lower `FullTriangular<f64>` → `DMatrix<f64>`;
2. upper `FullTriangular<f64>` → `DMatrix<f64>`;
3. `Complex64`;
4. `to_dmatrix()` preserves the source;
5. `into_dmatrix()` returns correct data;
6. `DMatrix` → lower `FullTriangular`;
7. `DMatrix` → upper `FullTriangular`;
8. non-square rejection;
9. dimensions:
   - `0`;
   - `1`;
   - `2`;
   - `3`;
   - odd and even larger dimensions;
10. exact column-major ordering.

Use coordinate-encoded entries so row/column transposition errors are obvious.

Example value:

```text
value(i, j) = 100*j + i
```

for real matrices.

For complex matrices use distinct real and imaginary coordinate encodings.

## Feature isolation tests

Run:

```bash
cargo check --no-default-features
cargo test --no-default-features
cargo check --features nalgebra-interop
cargo test --features nalgebra-interop
```

Verify nalgebra is not present in the normal dependency tree when the feature is disabled:

```bash
cargo tree --no-default-features
```

Verify it is present when enabled:

```bash
cargo tree --features nalgebra-interop
```

## Documentation

Document:

- that nalgebra interoperability is optional;
- that conversions allocate full `n × n` storage;
- that they are not zero-copy views;
- column-major compatibility;
- `to_dmatrix()` versus `into_dmatrix()` allocation behavior;
- selected triangle semantics.

Do not yet add packed lower/upper conversions; those belong to Prompt 02.

## Validation

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo check --all-targets --no-default-features
cargo test --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo test --features nalgebra-interop
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
git diff --check
```

Inspect:

```bash
git diff master...HEAD --stat
git diff master...HEAD
```

## Commit and PR

Commit:

```text
Add nalgebra interoperability foundation
```

Push:

```text
agent/nalgebra-interop-foundation
```

Open a draft PR targeting `master`.

The PR description must state:

- nalgebra remains optional;
- feature name;
- dependency impact;
- `FullTriangular` conversion APIs;
- ownership and allocation behavior;
- tests and commands run.

Only after the final diff is reviewed, checks pass, the PR is open, and no more commits are planned, finish with this exact standalone line:

**Safe to rebase and merge.**
