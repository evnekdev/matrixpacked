// lib.rs

//! Triangularly packed matrix representations for efficient use with LAPACK algorithms.

pub mod error;
pub mod scalar;
pub mod storage;
mod formatting;

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
