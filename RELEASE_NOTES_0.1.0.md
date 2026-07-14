# matrixpacked 0.1.0

`matrixpacked 0.1.0` is the first public release of a Rust library for square
structured matrices stored directly in traditional BLAS/LAPACK packed-column
formats.

## Key capabilities

- Packed lower and upper triangular, symmetric, Hermitian, and SPD/HPD matrix
  families.
- Owned storage plus immutable and mutable zero-copy views of packed buffers.
- Arithmetic, multiplication, rank updates, factorizations, solves, inverses,
  condition estimates, equilibration, refinement, and diagnostics.
- Simple and expert solve drivers.
- Standard and generalized symmetric/Hermitian eigensolvers, selected-spectrum
  drivers, reductions, and packed/full/RFP format conversions.
- Optional nalgebra conversion and validation support.
- Reproducible unit, integration, oracle, property, documentation, and
  native-provider test coverage.

## Installation

For packed storage and provider-free APIs:

```toml
[dependencies]
matrixpacked = "0.1"
```

Numerical operations call BLAS/LAPACK symbols. Select one supported bundled
provider, or arrange compatible native symbols in the final application.

Linux with static OpenBLAS:

```toml
[dependencies]
matrixpacked = { version = "0.1", features = ["openblas-static"] }
```

Windows with sequential LP64 Intel MKL:

```toml
[dependencies]
matrixpacked = { version = "0.1", features = ["intel-mkl-static"] }
```

Do not enable two native providers in the same resolved dependency graph.

## Optional nalgebra interoperability

Enable `nalgebra-interop` to convert owned `nalgebra::DMatrix` values, extract
explicit triangles, validate symmetric or Hermitian structure with tolerances,
and validate SPD/HPD matrices through Cholesky factorization:

```toml
[dependencies]
matrixpacked = { version = "0.1", features = ["nalgebra-interop", "openblas-static"] }
```

Choose the provider feature appropriate for the target platform. Packed storage
cannot be exposed as a zero-copy rectangular nalgebra view.

## Documentation

- [API documentation](https://docs.rs/matrixpacked/0.1.0/matrixpacked/)
- [Repository README](https://github.com/evnekdev/matrixpacked/blob/master/README.md)
- [Nalgebra interoperability](https://github.com/evnekdev/matrixpacked/blob/master/NALGEBRA_INTEROP.md)
- [Testing and reproducibility](https://github.com/evnekdev/matrixpacked/blob/master/TESTING.md)
- [Packed BLAS/LAPACK coverage](https://github.com/evnekdev/matrixpacked/blob/master/PACKED_LAPACK_FUNCTIONS.md)
- [Supported operations](https://github.com/evnekdev/matrixpacked/blob/master/OPERATIONS.md)

## Known limitations

- Linked numerical operations require a native BLAS/LAPACK provider. The crate
  deliberately selects none by default.
- The repository-supported bundled-provider workflow uses OpenBLAS on Linux and
  Intel MKL on Windows. Non-vcpkg OpenBLAS source builds are not assumed to work
  on Windows.
- Triangular nalgebra conversions currently follow a LAPACK conversion path and
  therefore require a provider. GitHub issue
  [#41](https://github.com/evnekdev/matrixpacked/issues/41) tracks a
  backend-independent implementation; the current behavior is not a correctness
  defect.
- As a pre-1.0 crate, the API may evolve in later minor releases. Incompatible
  changes will be documented in the changelog.

## Minimum supported Rust version

The minimum supported Rust version is Rust 1.89 for both the core crate and the
optional `nalgebra-interop` feature.

This is the first public release of `matrixpacked`.
