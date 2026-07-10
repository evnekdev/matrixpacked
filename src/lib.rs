// lib.rs

//! Triangually packed matrix representations for efficient use with LAPACK algorithms.

pub mod scalar;
pub mod storage;
pub mod error;

pub mod lower;
pub mod upper;
pub mod symmetric;
pub mod spd;
pub mod hermitian;

