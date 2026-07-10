// packedmatrix::scalar.rs

use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::fmt::Debug;

mod private {
	pub trait Sealed {}
	impl Sealed for f32 {}
	impl Sealed for f64 {}
	impl Sealed for Complex32 {]
	impl Sealed for Complex64 {}
}

/// Scalar types directly supported by conventional LAPACK routines.
///
/// LAPACK families:
///
/// - `s*`: `f32`
/// - `d*`: `f64`
/// - `c*`: `Complex32`
/// - `z*`: `Complex64`
pub trait LapackScalar:
    private::Sealed + Copy + Debug + Zero + Send + Sync + 'static
{
    /// The corresponding real scalar type.
    type Real: Copy + Debug + Zero + Send + Sync + 'static;
}

impl LapackScalar for f32 {
    type Real = f32;
}

impl LapackScalar for f64 {
    type Real = f64;
}

impl LapackScalar for Complex32 {
    type Real = f32;
}

impl LapackScalar for Complex64 {
    type Real = f64;
}