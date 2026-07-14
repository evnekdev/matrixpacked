# Prompt 05 — Add comprehensive nalgebra conversion tests

Work on:

https://github.com/evnekdev/matrixpacked

## Starting conditions

Start after Prompt 04 is merged.

Create:

```text
agent/nalgebra-conversion-tests
```

## Goal

Build comprehensive deterministic and property-based tests for every nalgebra interoperability conversion.

This PR should primarily add tests and fix concrete defects discovered by those tests.

Do not add unrelated APIs.

## Complete conversion matrix

Audit and test every supported direction:

| matrixpacked type | to `DMatrix` | extraction from `DMatrix` | strict validated conversion |
|---|---:|---:|---:|
| `FullTriangular` | yes | yes | shape validation |
| `PackedLower` | yes | yes | yes |
| `PackedUpper` | yes | yes | yes |
| `PackedSymmetric` | yes | yes | yes |
| `PackedHermitian` | yes | yes | yes |
| `PackedSPD` | yes | yes | yes, including PD |

Test owned, view, and mutable-view sources for packed → nalgebra conversions.

## Independent oracle rule

Expected matrices and packed buffers must be constructed independently.

Do not compute expected output by invoking another matrixpacked conversion.

Use explicit formulas and nalgebra indexing.

Round-trip tests are necessary but not sufficient.

## Deterministic dimensions

Use:

```text
0, 1, 2, 3, 4, 5, 8, 9
```

Cover odd and even dimensions.

## Scalar coverage

Use:

```text
f32
f64
Complex32
Complex64
```

for every valid matrix family.

## Exact storage-layout tests

Use coordinate-encoded values.

For traditional lower packed storage independently verify:

```text
a00,
a10, a20, ...,
a11, a21, ...,
...
```

For upper packed storage independently verify:

```text
a00,
a01, a11,
a02, a12, a22,
...
```

Confirm reverse nalgebra conversion produces the exact expected packed sequence.

## Round-trip tests

Test:

```text
packed -> DMatrix -> validated packed
packed -> DMatrix -> extracted packed
DMatrix -> packed -> DMatrix
FullTriangular -> DMatrix -> FullTriangular
```

Use exact equality where no floating-point arithmetic occurs.

Use tolerance only for validation-specific noisy inputs.

## Structure tests

### Lower/upper

- structural zeros;
- ignored triangle extraction;
- strict rejection;
- diagonal preservation.

### Symmetric

- real symmetric;
- complex symmetric;
- no accidental conjugation.

### Hermitian

- conjugate mirroring;
- real diagonal;
- phase-rich complex values.

### SPD/HPD

- positive-definite acceptance;
- indefinite rejection;
- semidefinite rejection;
- ill-scaled positive-definite matrices.

## Property-based testing

Use the existing `proptest` infrastructure.

Properties:

1. independent structured full matrix → strict packed → full equals source within tolerance;
2. arbitrary packed storage → full → strict packed preserves packed storage;
3. lower conversion always produces zero logical upper triangle;
4. upper conversion always produces zero logical lower triangle;
5. symmetric output remains symmetric;
6. complex-symmetric output does not become Hermitian accidentally;
7. Hermitian output satisfies:
   ```text
   A ≈ Aᴴ
   ```
8. accepted SPD/HPD matrices succeed under nalgebra Cholesky;
9. `to_dmatrix()` does not mutate source views;
10. extraction uses only the documented triangle.

Keep dimensions moderate:

```text
0..12
```

Use deterministic proptest configuration or print reproducible seeds.

## Allocation and ownership tests

Where practical, verify:

- `FullTriangular::into_dmatrix()` consumes the original buffer;
- `to_dmatrix()` leaves source available;
- packed structured conversion allocates full storage;
- no API claims zero-copy behavior.

Do not rely on unstable allocator internals.

Pointer-reuse testing is optional and should not be brittle.

## Feature-gating tests

Run:

```bash
cargo check --no-default-features
cargo test --no-default-features
cargo check --features nalgebra-interop
cargo test --features nalgebra-interop
```

Ensure nalgebra examples and tests do not compile when the feature is absent unless correctly gated.

## Native-backend independence

Pure nalgebra conversion tests should not require an OpenBLAS or MKL feature.

If triangular conversions currently call `xTPTTR`/`xTRTTP`, those tests may require final native linking.

Evaluate whether this makes the optional interoperability feature unnecessarily backend-dependent.

If so, do not silently rewrite the production conversion architecture in this test PR.

Document the limitation and create a follow-up issue, or make only a narrowly scoped fix agreed by the existing design.

## Documentation status

Update any conversion coverage table or create:

```text
NALGEBRA_INTEROP.md
```

only if a concise test/feature matrix adds value.

Do not duplicate full API docs.

## Validation and PR

Commit:

```text
Test nalgebra interoperability
```

Branch:

```text
agent/nalgebra-conversion-tests
```

PR title:

```text
Test nalgebra interoperability
```

The PR description must list:

- types;
- directions;
- scalars;
- dimensions;
- deterministic tests;
- property tests;
- backend requirements.

Finish only with:

**Safe to rebase and merge.**
