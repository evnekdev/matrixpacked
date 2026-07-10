// lib.rs

//! Triangularly packed matrix representations with direct BLAS/LAPACK packed-format operations.
//!
//! Enable `openblas-static` to bundle an OpenBLAS provider, or link another compatible
//! BLAS/LAPACK implementation in the final application.

pub mod error;
pub mod scalar;
pub mod storage;
mod formatting;
mod backend;
pub mod factorization;
mod arithmetic;

#[cfg(feature = "openblas-static")]
use openblas_src as _;

pub mod hermitian;
pub mod lower;
pub mod spd;
pub mod symmetric;
pub mod upper;

pub use error::PackedMatrixError;
pub use hermitian::{PackedHermitian, PackedHermitianView, PackedHermitianViewMut};
pub use lower::{PackedLower, PackedLowerView, PackedLowerViewMut};
pub use scalar::LapackScalar;
pub use spd::{PackedSPD, PackedSPDView, PackedSPDViewMut};
pub use symmetric::{PackedSymmetric, PackedSymmetricView, PackedSymmetricViewMut};
pub use upper::{PackedUpper, PackedUpperView, PackedUpperViewMut};

pub use factorization::{PackedCholesky, PackedCholeskyViewMut, PackedSymmetricFactor, PackedSymmetricFactorViewMut, PackedHermitianFactor, PackedHermitianFactorViewMut};
