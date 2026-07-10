// lib.rs

//! Triangually packed matrix representations for efficient use with LAPACK algorithms.

pub mod error;
pub mod scalar;
pub mod storage;

pub mod hermitian;
pub mod lower;
pub mod spd;
pub mod symmetric;
pub mod upper;

pub use error::PackedMatrixError;

pub use lower::{PackedLower, PackedLowerView, PackedLowerViewMut};

pub use scalar::LapackScalar;
