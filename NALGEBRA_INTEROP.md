# Nalgebra interoperability

Enable the optional integration with:

```toml
matrixpacked = { version = "0.1", features = ["nalgebra-interop"] }
```

This adds conversions to and from owned `nalgebra::DMatrix` values. It does not
make nalgebra mandatory for ordinary matrixpacked users.

## Conversion, not viewing

Traditional packed storage contains `n(n+1)/2` values in variable-length
columns. Nalgebra's rectangular stride model describes full storage with `n²`
values and cannot directly view that layout. A `1000 × 1000` `f64` matrix, for
example, grows from 500,500 packed values (about 3.8 MiB) to 1,000,000 full
values (about 7.6 MiB). Consequently, every packed conversion allocates.
`FullTriangular::into_dmatrix` can instead move its already-full owned buffer.

## Examples

Expand a packed lower triangle (requires a linked LAPACK provider):

```rust
let packed = PackedLower::from_vec(2, vec![1.0_f64, 2.0, 3.0])?;
let dense = packed.to_dmatrix()?;
```

Strictly validate symmetry:

```rust
let packed = PackedSymmetric::try_from_dmatrix(
    &dense,
    ConversionTolerance::new(1.0e-12, 1.0e-12),
)?;
```

Extraction is intentionally different: this keeps the lower triangle and
ignores every value above the diagonal.

```rust
let packed = PackedLower::from_lower_triangle(&dense)?;
```

For an SPD/HPD claim, validate both structure and positive definiteness with
nalgebra Cholesky:

```rust
let packed = PackedSPD::try_from_dmatrix(&dense, ConversionTolerance::default())?;
```

## Extraction versus validation

Methods named `from_lower_triangle`, `from_upper_triangle`, or
`from_lower_triangle_unchecked_structure` extract one triangle and deliberately
discard the other. Methods named `try_from_dmatrix` use the complete matrix as
validation evidence. They report non-square inputs, invalid tolerances, and the
first structural mismatch; SPD/HPD validation can additionally report a failed
Cholesky test. No constructor averages opposite entries.

## Complex symmetric is not Hermitian

Suppose the stored lower off-diagonal entry is `2 + 3i`. A complex symmetric
matrix mirrors it unchanged, so the corresponding upper entry is also `2 + 3i`.
A Hermitian matrix conjugates it, so the upper entry is `2 - 3i`. Hermitian and
complex HPD conversions also normalize the diagonal to the LAPACK convention of
a real diagonal.

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
