// packedmatrix::scalar.rs

//use num_complex::{Complex};
use num_traits::Zero;
use std::fmt::Debug;

mod private {
	pub trait Sealed {}
	impl Sealed for f32 {}
	impl Sealed for f64 {}
	impl Sealed for num_complex::Complex<f32> {}
	impl Sealed for num_complex::Complex<f64> {}
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

impl LapackScalar for num_complex::Complex<f32> {
    type Real = f32;
}

impl LapackScalar for num_complex::Complex<f64> {
    type Real = f64;
}