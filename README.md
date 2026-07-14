# matrixpacked

[![CI](https://github.com/evnekdev/matrixpacked/actions/workflows/ci.yml/badge.svg)](https://github.com/evnekdev/matrixpacked/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/matrixpacked.svg)](https://crates.io/crates/matrixpacked)
[![docs.rs](https://img.shields.io/docsrs/matrixpacked)](https://docs.rs/matrixpacked/0.1.0/matrixpacked/)

Packed triangular, symmetric, Hermitian, and positive-definite matrices backed
directly by traditional BLAS/LAPACK packed-column storage.

The crate stores `n(n+1)/2` scalars, supports owned matrices and zero-copy
borrowed views, and exposes packed matrix-vector operations, factorizations,
solves, refinement, diagnostics, eigensolvers, and format conversions.

```rust
use matrixpacked::{PackedMatrixError, PackedSPD};

fn main() -> Result<(), PackedMatrixError> {
    let a = PackedSPD::from_vec(2, vec![4.0_f64, 1.0, 3.0])?;
    let factor = a.cholesky()?;
    let x = factor.solve_vector(&[6.0, 7.0])?;
    assert_eq!(x.len(), 2);
    Ok(())
}
```

Running numerical operations such as this solve requires a native BLAS/LAPACK
provider in the final application.

## Installation

For packed storage and provider-free APIs:

```toml
[dependencies]
matrixpacked = "0.1"
```

Choose one native provider below when using linked numerical operations.

## Native backend

`blas` and `lapack` are bindings, so numerical executables need a native
provider. The crate selects none by default. Enable the repository-supported
bundled provider for your platform, or link compatible symbols in the final
application:

```toml
# Linux
matrixpacked = { version = "0.1", features = ["openblas-static"] }

# Windows
matrixpacked = { version = "0.1", features = ["intel-mkl-static"] }
```

Select only one bundled provider. The supported Windows workflow uses MKL; the
non-vcpkg OpenBLAS source build is not assumed to work there.

Cargo unifies features across the dependency graph. If different dependencies
enable both bundled provider features, both providers are resolved; the crate
does not reject that combination at compile time. Linking two native providers
can produce conflicting symbols and is not a supported configuration, so the
final application should ensure that only one provider feature is selected.

## Optional nalgebra support

Enable `nalgebra-interop` for owned `nalgebra::DMatrix` conversion, explicit
triangle extraction, tolerance-aware structural validation, and nalgebra
Cholesky-backed SPD/HPD validation:

```toml
matrixpacked = { version = "0.1", features = ["nalgebra-interop"] }
```

Packed storage cannot be exposed as a zero-copy rectangular nalgebra view.
Structured conversion is pure Rust; triangular packed/full conversion currently
requires a native LAPACK provider.

## Supported Rust version

The minimum supported Rust version for `matrixpacked 0.1.0` is Rust 1.89.
This version covers the core crate and the optional `nalgebra-interop` feature.

## Development commands

```bash
# Provider-free compilation and documentation
cargo check --all-targets --no-default-features
cargo doc --no-deps --features nalgebra-interop

# Linux linked suite
cargo test --all-targets --features openblas-static

# Windows linked suite
cargo test --all-targets --features intel-mkl-static
```

Runner scripts are `scripts_run_lapack_examples.{sh,bat}` and
`scripts_run_nonlapack_examples.{sh,bat}`. The batch scripts use
`--intel-mkl-static`; the shell LAPACK runner accepts `--openblas-static`.

## Stability

`matrixpacked` is pre-1.0. Minor releases may include API changes while the
interface evolves; user-visible changes will be recorded in the changelog.

## Guides and status

- [Complete crate guide](https://docs.rs/matrixpacked/latest/matrixpacked/)
- [Nalgebra interoperability](https://github.com/evnekdev/matrixpacked/blob/master/NALGEBRA_INTEROP.md)
- [Testing and reproducibility](https://github.com/evnekdev/matrixpacked/blob/master/TESTING.md)
- [LAPACK routine coverage](https://github.com/evnekdev/matrixpacked/blob/master/PACKED_LAPACK_FUNCTIONS.md)
- [Runnable example coverage](https://github.com/evnekdev/matrixpacked/blob/master/EXAMPLE_COVERAGE.md)
- [Non-LAPACK example coverage](https://github.com/evnekdev/matrixpacked/blob/master/NON_LAPACK_EXAMPLES.md)
- [Operator surface](https://github.com/evnekdev/matrixpacked/blob/master/OPERATIONS.md)
- [Changelog](https://github.com/evnekdev/matrixpacked/blob/master/CHANGELOG.md)

## License

Licensed under either of
[Apache License, Version 2.0](https://github.com/evnekdev/matrixpacked/blob/master/LICENSE-APACHE)
or the [MIT license](https://github.com/evnekdev/matrixpacked/blob/master/LICENSE-MIT)
at your option.
