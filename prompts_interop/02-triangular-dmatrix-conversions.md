# Prompt 02 — Add packed lower/upper and nalgebra conversions

Work on:

https://github.com/evnekdev/matrixpacked

## Starting conditions

Start only after Prompt 01 has been merged.

1. Fetch remote changes.
2. Switch to `master`.
3. Pull using fast-forward only.
4. Confirm Prompt 01 is present on `master`.
5. Create:

```text
agent/nalgebra-triangular-conversions
```

## Goal

Implement optional nalgebra interoperability for:

```text
PackedLower<T, S>
PackedUpper<T, S>
```

Support conversion to and from:

```text
nalgebra::DMatrix<T>
```

Use the existing LAPACK-backed `xTPTTR` and `xTRTTP` conversion paths where they are the correct storage-format conversion.

## Architectural rule

Distinguish:

### LAPACK format conversion

```text
traditional packed TP ↔ full triangular TR
```

Use existing specialized LAPACK routines:

```text
xTPTTR
xTRTTP
```

### nalgebra container conversion

```text
FullTriangular<T> ↔ DMatrix<T>
```

Use the conversion APIs introduced in Prompt 01.

The preferred pipeline is:

```text
PackedLower
  -> xTPTTR
  -> FullTriangular
  -> DMatrix
```

and:

```text
DMatrix
  -> validated/extracted FullTriangular
  -> xTRTTP
  -> PackedLower
```

Do not duplicate packed-index formulas in production code unless the existing LAPACK path cannot satisfy the API.

## Packed to nalgebra

Implement feature-gated methods for every packed storage backend:

```rust
impl<T, S> PackedLower<T, S>
where
    T: PackedFormatConversion + nalgebra::Scalar,
    S: PackedStorage<T>,
{
    pub fn to_dmatrix(&self) -> Result<DMatrix<T>, PackedMatrixError>
    where
        T: Clone;
}
```

and similarly for `PackedUpper`.

Because `xTPTTR` writes full triangular storage:

- the selected triangle contains matrix values;
- the opposite triangle should contain structural zeros if `FullTriangular` guarantees this;
- ensure the resulting `DMatrix` is a logical triangular matrix.

Do not mirror triangular entries.

### Allocation behavior

Document that this conversion allocates `n * n` elements.

There is no zero-copy `DMatrixView` over traditional packed storage.

## Nalgebra to packed: two policy variants

A full `DMatrix<T>` may contain nonzero entries outside the selected triangle.

Provide two clearly different APIs.

### Strict validation

```rust
pub fn try_from_dmatrix(
    matrix: &DMatrix<T>,
    tolerance: ...
) -> Result<PackedLower<T>, PackedMatrixError>;
```

This method:

- requires square shape;
- requires entries outside the selected triangle to be zero within tolerance;
- rejects structurally invalid matrices;
- packs the selected triangle using `xTRTTP`.

Likewise for upper triangular.

### Explicit extraction

```rust
pub fn from_lower_triangle(matrix: &DMatrix<T>) -> Result<PackedLower<T>, PackedMatrixError>;

pub fn from_upper_triangle(matrix: &DMatrix<T>) -> Result<PackedUpper<T>, PackedMatrixError>;
```

These methods:

- require square shape;
- intentionally ignore the opposite triangle;
- do not pretend to validate triangular structure;
- document the discarded values explicitly.

Use names that make extraction obvious.

Do not make `try_from_dmatrix()` silently discard half the matrix.

## Tolerance design

Prompt 04 will introduce the full reusable tolerance API.

For this PR, choose one of these approaches:

1. accept explicit absolute tolerance parameters using `T::Real`;
2. implement only explicit extraction now and defer strict approximate validation to Prompt 04;
3. implement exact-zero strict validation only for now and state that approximate validation comes later.

Preferred approach:

- implement extraction methods now;
- add strict tolerance-aware methods in Prompt 04.

Do not design a temporary public API that will need to be removed immediately.

## Scalar families

Support:

```text
f32
f64
Complex32
Complex64
```

using the existing `PackedFormatConversion` backend.

For complex triangular matrices, zero validation uses magnitude, not separate exact real/imaginary checks.

## Ownership and views

Conversions from all of these should work:

```text
PackedLower<T>
PackedLowerView<'_, T>
PackedLowerViewMut<'_, T>
PackedUpper<T>
PackedUpperView<'_, T>
PackedUpperViewMut<'_, T>
```

Generic methods on `S: PackedStorage<T>` should cover them naturally.

Converting a `DMatrix` into packed form returns an owned packed matrix because the layouts are incompatible and require allocation.

Do not attempt zero-copy views.

## Tests

Add:

```text
tests/nalgebra_triangular_conversions.rs
```

Test all scalar families.

Required cases:

1. packed lower → `DMatrix`;
2. packed upper → `DMatrix`;
3. `DMatrix` lower extraction → packed;
4. `DMatrix` upper extraction → packed;
5. packed → matrix → packed exact round trip;
6. matrix triangle extraction → packed → matrix;
7. opposite triangle becomes zero in resulting logical triangular matrix;
8. source view remains unchanged;
9. mutable view conversion does not mutate storage;
10. dimensions:
    - `0`;
    - `1`;
    - `2`;
    - `3`;
    - `4`;
    - `7`;
11. complex values;
12. exact packed storage ordering after reverse conversion;
13. non-square rejection;
14. feature-disabled compilation.

Use independent expected full matrices rather than only round-trip assertions.

A pair of mutually inverse bugs can pass a round-trip test.

## Examples

Add focused user examples:

```text
examples/nalgebra_lower_conversion.rs
examples/nalgebra_upper_conversion.rs
```

Gate examples with the feature through `required-features` entries in `Cargo.toml` if necessary:

```toml
[[example]]
name = "nalgebra_lower_conversion"
required-features = ["nalgebra-interop"]
```

Do not cause ordinary `cargo check --examples` without the feature to fail.

## Documentation

Explain:

- traditional packed versus full triangular storage;
- use of LAPACK `xTPTTR` and `xTRTTP`;
- allocation cost;
- extraction versus validation;
- no zero-copy views;
- column-major nalgebra storage.

Update `PACKED_LAPACK_FUNCTIONS.md` only if API status wording needs clarification.

## Validation

Run all feature-free and feature-enabled checks.

Also run:

```bash
cargo run --example nalgebra_lower_conversion --features nalgebra-interop
cargo run --example nalgebra_upper_conversion --features nalgebra-interop
```

Use the repository’s native backend feature if final linking requires one.

## Commit and PR

Commit:

```text
Add nalgebra triangular conversions
```

Branch:

```text
agent/nalgebra-triangular-conversions
```

PR title:

```text
Add nalgebra triangular conversions
```

Finish only when complete with:

**Safe to rebase and merge.**
