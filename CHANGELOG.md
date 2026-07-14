# Changelog

All notable user-visible changes to `matrixpacked` are documented in this file.

## [Unreleased]

## [0.1.0]

Initial public release.

### Added

- Packed lower- and upper-triangular matrices, real and complex symmetric
  matrices, complex Hermitian matrices, and SPD/HPD matrices.
- Owned matrices, immutable borrowed views, and mutable borrowed views over
  traditional packed-column storage.
- Arithmetic, packed matrix-vector multiplication, triangular operations, and
  symmetric or Hermitian rank-1 and rank-2 updates where supported.
- Packed factorizations, simple and reusable solves, explicit inverses,
  reciprocal condition estimates, iterative refinement, equilibration, and
  expert solve reports.
- Standard and generalized symmetric or Hermitian eigensolvers, including
  basic, divide-and-conquer, and selected-spectrum drivers.
- Tridiagonal and generalized-problem reductions, transformation generation and
  application, and packed/full/RFP format conversions.
- Factor diagnostics including inertia, determinant, log-determinant, and
  triangular singularity checks.
- Optional `nalgebra-interop` conversions and structural validation for dynamic
  matrices.
- Unit, integration, oracle, property, edge-case, documentation, and
  provider-linked test coverage, plus guides for layouts, operations,
  interoperability, native providers, and reproducibility.

### Known limitations

- Numerical operations that call BLAS or LAPACK require the final application
  to link a compatible native provider; the crate selects none by default.
- The supported bundled-provider guidance uses OpenBLAS on Linux and sequential
  LP64 Intel MKL on Windows. Non-vcpkg OpenBLAS source builds are not assumed to
  work on Windows.
- Triangular nalgebra conversions currently use LAPACK packed/full conversion
  routines and therefore require a native provider. GitHub issue
  [#41](https://github.com/evnekdev/matrixpacked/issues/41) tracks making those
  conversions backend-independent; this is a documented provider dependency,
  not a correctness defect.
- This is a pre-1.0 release. Minor releases may contain API changes while the
  interface evolves, with changes documented here.

[Unreleased]: https://github.com/evnekdev/matrixpacked/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/evnekdev/matrixpacked/releases/tag/v0.1.0
