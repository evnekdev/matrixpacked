// lib.rs

// Established LAPACK FFI signatures and legacy expression style are intentionally
// retained; CI denies every warning outside this explicit compatibility set.
#![allow(
    private_bounds,
    dead_code,
    clippy::byte_char_slices,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::needless_return,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::useless_vec
)]

//! Triangularly packed matrix representations with direct BLAS/LAPACK packed-format operations.
//!
//! Enable `openblas-static` to bundle an OpenBLAS provider, use `intel-mkl-static`
//! for a Windows-compatible bundled provider, or link another compatible BLAS/LAPACK
//! implementation in the final application.
//!
//! # Nalgebra interoperability
//!
//! The optional `nalgebra-interop` feature adds owned conversions to and from
//! [`nalgebra::DMatrix`]:
//!
//! ```toml
//! matrixpacked = { version = "0.1", features = ["nalgebra-interop"] }
//! ```
//!
//! Interoperability is conversion, not viewing. Traditional packed columns have
//! different lengths, which nalgebra's rectangular stride model cannot describe.
//! Packed storage uses `n(n+1)/2` scalar values; a full matrix uses `n²`, so packed
//! conversions allocate full storage. [`FullTriangular::into_dmatrix`] is the one
//! exception that can move an already-owned `n × n` column-major buffer directly.
//!
//! Convert a packed lower triangle to a nalgebra matrix (this path uses LAPACK
//! `xTPTTR`, so the final binary must link a BLAS/LAPACK provider):
//!
//! ```no_run
//! # #[cfg(feature = "nalgebra-interop")]
//! # fn example() -> Result<(), matrixpacked::PackedMatrixError> {
//! use matrixpacked::PackedLower;
//! let packed = PackedLower::from_vec(2, vec![1.0_f64, 2.0, 3.0])?;
//! let dense = packed.to_dmatrix()?;
//! assert_eq!(dense[(1, 0)], 2.0);
//! # Ok(()) }
//! ```
//!
//! Strict constructors validate the complete matrix using an explicit absolute
//! and relative tolerance:
//!
//! ```
//! # #[cfg(feature = "nalgebra-interop")]
//! # fn example() -> Result<(), matrixpacked::PackedMatrixError> {
//! use matrixpacked::{ConversionTolerance, PackedSymmetric};
//! use nalgebra::DMatrix;
//! let dense = DMatrix::from_row_slice(2, 2, &[2.0_f64, 1.0, 1.0, 3.0]);
//! let packed = PackedSymmetric::try_from_dmatrix(
//!     &dense,
//!     ConversionTolerance::new(1.0e-12, 1.0e-12),
//! )?;
//! assert_eq!(packed.dimension(), 2);
//! # Ok(()) }
//! ```
//!
//! Extraction names deliberately ignore the opposite triangle; use a
//! `try_from_dmatrix` constructor when that triangle is evidence to validate:
//!
//! ```no_run
//! # #[cfg(feature = "nalgebra-interop")]
//! # fn example() -> Result<(), matrixpacked::PackedMatrixError> {
//! use matrixpacked::PackedLower;
//! use nalgebra::DMatrix;
//! let dense = DMatrix::from_row_slice(2, 2, &[1.0_f64, 99.0, 2.0, 3.0]);
//! let packed = PackedLower::from_lower_triangle(&dense)?; // ignores 99.0
//! assert_eq!(packed.as_slice(), &[1.0, 2.0, 3.0]);
//! # Ok(()) }
//! ```
//!
//! Positive-definite validation is a pure-Rust nalgebra Cholesky path after
//! structural validation:
//!
//! ```
//! # #[cfg(feature = "nalgebra-interop")]
//! # fn example() -> Result<(), matrixpacked::PackedMatrixError> {
//! use matrixpacked::{ConversionTolerance, PackedSPD};
//! use nalgebra::DMatrix;
//! let dense = DMatrix::from_row_slice(2, 2, &[4.0_f64, 1.0, 1.0, 3.0]);
//! let packed = PackedSPD::try_from_dmatrix(&dense, ConversionTolerance::default())?;
//! assert_eq!(packed.dimension(), 2);
//! # Ok(()) }
//! ```
//!
//! Complex symmetric and Hermitian matrices are distinct. If the lower entry is
//! `2 + 3i`, symmetric expansion puts `2 + 3i` above the diagonal, while
//! Hermitian expansion puts its conjugate `2 - 3i` there. Structured expansion,
//! tolerance checks, and nalgebra Cholesky are implemented in pure Rust. Only the
//! traditional triangular packed/full paths currently use specialized LAPACK
//! `xTPTTR`/`xTRTTP` format-conversion routines.

mod arithmetic;
mod backend;
mod conversions;
mod diagnostics;
mod eigen;
mod equilibration;
pub mod error;
mod expert_solve;
pub mod factorization;
mod formatting;
mod generalized_reduction;
#[cfg(feature = "nalgebra-interop")]
mod nalgebra_interop;
mod norms;
mod rank_updates;
pub mod scalar;
mod simple_solve;
pub mod storage;
pub mod triangular;
mod tridiagonal;

pub use conversions::{
    FullTriangular, RectangularFullPacked, RectangularFullPackedView, RectangularFullPackedViewMut,
    RfpTranspose, Triangle,
};
pub use diagnostics::{Inertia, SignedLogDet};
pub use equilibration::Equilibration;
pub use expert_solve::{EquilibrationMode, ExpertSolveOptions, ExpertSolveResult};
pub use tridiagonal::{
    ApplySide, HermitianPackedTridiagonal, OrthogonalOperation, SymmetricPackedTridiagonal,
    UnitaryOperation,
};

#[cfg(feature = "openblas-static")]
use openblas_src as _;

#[cfg(feature = "intel-mkl-static")]
use intel_mkl_src as _;

pub mod hermitian;
pub mod lower;
pub mod spd;
pub mod symmetric;
pub mod upper;

pub use eigen::{
    EigenDecomposition, EigenRange, Eigenvectors, GeneralizedEigenproblem,
    SelectedEigenDecomposition,
};
pub use error::PackedMatrixError;
pub use hermitian::{PackedHermitian, PackedHermitianView, PackedHermitianViewMut};
pub use lower::{PackedLower, PackedLowerView, PackedLowerViewMut};
#[cfg(feature = "nalgebra-interop")]
pub use nalgebra_interop::{ConversionTolerance, DefaultConversionTolerance};
pub use scalar::LapackScalar;
pub use spd::{PackedSPD, PackedSPDView, PackedSPDViewMut};
pub use symmetric::{PackedSymmetric, PackedSymmetricView, PackedSymmetricViewMut};
pub use triangular::{ConditionNorm, Diagonal, MatrixNorm, RefinementReport, Transpose};
pub use upper::{PackedUpper, PackedUpperView, PackedUpperViewMut};

pub use factorization::{
    PackedCholesky, PackedCholeskyViewMut, PackedHermitianFactor, PackedHermitianFactorViewMut,
    PackedSymmetricFactor, PackedSymmetricFactorViewMut,
};
