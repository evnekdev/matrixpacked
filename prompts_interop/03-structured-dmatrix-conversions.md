# Prompt 03 — Add symmetric, Hermitian, and SPD nalgebra conversions

Work on:

https://github.com/evnekdev/matrixpacked

## Starting conditions

Start only after Prompt 02 is merged.

Create:

```text
agent/nalgebra-structured-conversions
```

## Goal

Implement optional nalgebra conversions for:

```text
PackedSymmetric<T, S>
PackedHermitian<T, S>
PackedSPD<T, S>
```

Support:

```text
f32
f64
Complex32
Complex64
```

according to each matrix family’s valid semantics.

## Important distinction

These logical matrices are not triangular.

Do not return only the stored half.

Expand the complete logical matrix:

### Symmetric

```text
A(j, i) = A(i, j)
```

No conjugation, including for complex symmetric matrices.

### Hermitian

```text
A(j, i) = conj(A(i, j))
```

The logical diagonal must be real.

### SPD/HPD

- real scalar: symmetric expansion;
- complex scalar: Hermitian expansion.

## Implementation choice

Use pure Rust for logical expansion to nalgebra.

Do not call `xTPTTR` merely to fill one triangle and then perform a second pass unless benchmarking or code reuse clearly justifies it.

Reasons:

- the result requires both triangles;
- direct packed traversal can fill both output entries in one pass;
- conversion should work without linking a native LAPACK provider;
- FFI overhead is unnecessary for a straightforward memory transformation.

Use a single efficient packed traversal.

Do not repeatedly call a potentially nontrivial public `(i, j)` accessor inside a full `n²` double loop if a one-pass packed traversal is available.

## Packed to nalgebra APIs

Implement:

```rust
pub fn to_dmatrix(&self) -> DMatrix<T>
where
    T: Clone;
```

for all relevant types.

Return `DMatrix<T>` directly if conversion cannot fail after construction invariants are established.

Use `Result` only if dimension overflow or another real runtime error remains possible.

### Owned consuming conversion

Consider:

```rust
pub fn into_dmatrix(self) -> DMatrix<T>;
```

For structured packed matrices this cannot reuse the packed `Vec<T>` directly because the result has `n²` elements.

It will allocate regardless.

Do not add `into_dmatrix()` merely to imply zero-copy reuse.

Add it only if consuming semantics are still useful and documented honestly.

## Nalgebra to packed extraction APIs

Add explicit extraction methods:

```rust
PackedSymmetric::from_lower_triangle(&matrix)
PackedHermitian::from_lower_triangle(&matrix)
PackedSPD::from_lower_triangle(&matrix)
```

Adapt names to the actual stored triangle used by each type.

Inspect whether these types always use lower traditional packed storage.

The extraction methods:

- require square shape;
- copy only the relevant triangle;
- intentionally ignore the opposite triangle;
- do not validate symmetry, Hermitian structure, or positive definiteness;
- must be documented as extraction.

If `PackedSPD` represents an intended numerical invariant, name unchecked construction clearly:

```rust
PackedSPD::from_lower_triangle_unchecked_structure(...)
```

Avoid Rust `unsafe` unless violating the invariant can cause memory unsafety.

Numerical invalidity alone normally does not justify `unsafe`.

## Strict validated conversions

Do not fully implement tolerance-based validation in this prompt unless Prompt 04 has already been merged out of order.

Prompt 04 will add:

```text
try_from_dmatrix
```

with reusable tolerance and optional positive-definiteness checks.

Ensure extraction API names will coexist cleanly with those future strict constructors.

## Complex semantic tests

Create matrices that distinguish:

```text
complex symmetric:
A(j,i) = A(i,j)

Hermitian:
A(j,i) = conj(A(i,j))
```

For example, use an off-diagonal value:

```text
2 + 3i
```

Then verify:

- complex symmetric opposite entry is `2 + 3i`;
- Hermitian opposite entry is `2 - 3i`.

Do not use only real-valued complex test data.

## Hermitian diagonal

Inspect the crate’s existing diagonal invariant.

The nalgebra output must contain real diagonal values.

If the packed input may store a small imaginary diagonal component, follow the type’s documented behavior:

- reject at construction;
- normalize;
- or ignore imaginary components.

Do not silently introduce a different convention in the conversion layer.

## Tests

Add:

```text
tests/nalgebra_structured_conversions.rs
```

Cover:

### Symmetric

- `f32`;
- `f64`;
- `Complex32`;
- `Complex64`;
- exact mirroring without conjugation.

### Hermitian

- `Complex32`;
- `Complex64`;
- conjugated mirroring;
- real diagonal.

### SPD/HPD

- real symmetric expansion;
- complex Hermitian expansion;
- no claim that extraction validates positive definiteness.

### Storage backends

- owned;
- immutable view;
- mutable view.

### Dimensions

```text
0, 1, 2, 3, 4, 7
```

### Independent expected values

Construct expected nalgebra matrices independently.

Do not rely solely on conversion round trips.

## Examples

Add:

```text
examples/nalgebra_symmetric_conversion.rs
examples/nalgebra_hermitian_conversion.rs
examples/nalgebra_spd_conversion.rs
```

Require the `nalgebra-interop` feature.

## Documentation

Document prominently:

- complex symmetric is not Hermitian;
- SPD/HPD extraction does not prove positive definiteness;
- conversion allocates full storage;
- views cannot be zero-copy;
- opposite triangle reconstruction rules.

## Validation and PR

Commit:

```text
Add nalgebra structured conversions
```

Branch:

```text
agent/nalgebra-structured-conversions
```

PR title:

```text
Add nalgebra structured conversions
```

Run all normal, feature-gated, Clippy, rustdoc, tests, and examples.

Finish only with:

**Safe to rebase and merge.**
