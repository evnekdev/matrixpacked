# Nalgebra interoperability status

The `nalgebra-interop` feature exposes owned `DMatrix` conversions. Packed
storage cannot be represented as a zero-copy `DMatrix` view, so packed-to-full
operations allocate. `FullTriangular::into_dmatrix` can move its owned full
buffer; `to_dmatrix` leaves the source available.

## Conversion and test matrix

| Type | To `DMatrix` | Extraction from `DMatrix` | Strict validation | Packed source forms tested |
|---|---|---|---|---|
| `FullTriangular` | yes | selected triangle | square shape | owned |
| `PackedLower` | yes | lower | opposite triangle approximately zero | owned, view, mutable view |
| `PackedUpper` | yes | upper | opposite triangle approximately zero | owned, view, mutable view |
| `PackedSymmetric` | yes | lower | symmetric, without complex conjugation | owned, view, mutable view |
| `PackedHermitian` | yes | lower | Hermitian and real diagonal | owned, view, mutable view |
| `PackedSPD` | yes | lower, unchecked structure | structured-only and Cholesky-validated PD | owned, view, mutable view |

Independent coordinate/indexing oracles cover `f32`, `f64`, `Complex32`, and
`Complex64` at dimensions 0, 1, 2, 3, 4, 5, 8, and 9. Exact tests verify
traditional lower and upper packed-column order, logical structural zeros,
diagonal preservation, symmetric versus Hermitian reconstruction, extraction,
strict conversion, round trips, and source non-mutation.

Interop-specific proptests use dimensions `0..12` and the shared deterministic
configuration in `tests/oracle/properties.rs`. `MATRIXPACKED_PROPTEST_SEED` and
`MATRIXPACKED_PROPTEST_CASES` can override the reproducible default seed and
case count.

## Native backend status

The tolerance-aware structured conversions and nalgebra Cholesky validation
are pure Rust. Compile-only checks work without a native provider:

```text
cargo check --no-default-features
cargo check --all-targets --no-default-features
cargo check --features nalgebra-interop
```

Traditional triangular packed/full conversion currently calls LAPACK
`xTPTTR`/`xTRTTP`. Test binaries exercising `PackedLower` or `PackedUpper`
therefore need `openblas-static` or `intel-mkl-static` at final link time:

```text
cargo test --features "nalgebra-interop,intel-mkl-static"
```

This limitation is tracked in
[#41](https://github.com/evnekdev/matrixpacked/issues/41). The test PR records
the behavior without changing the production conversion architecture.

The repository's broader `cargo test --no-default-features` command also links
existing LAPACK examples and therefore requires a native provider. This is
independent of nalgebra feature gating; `cargo check --all-targets` verifies
that nalgebra-only tests and examples are excluded when the feature is absent.
