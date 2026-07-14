// packedmatrix::scalar.rs

//use num_complex::{Complex};
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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
    private::Sealed
    + Copy
    + Debug
    + Zero
    + One
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// The corresponding real scalar type.
    type Real: Copy + Debug + Zero + Send + Sync + 'static + Add<Output = Self::Real>;

    /// Returns the complex conjugate, or the value unchanged for real scalars.
    fn conjugate(self) -> Self;

    /// Returns the squared absolute value as the corresponding real scalar.
    fn abs_squared(self) -> Self::Real;
}

impl LapackScalar for f32 {
    type Real = f32;
    fn conjugate(self) -> Self {
        self
    }
    fn abs_squared(self) -> Self::Real {
        self * self
    }
}

impl LapackScalar for f64 {
    type Real = f64;
    fn conjugate(self) -> Self {
        self
    }
    fn abs_squared(self) -> Self::Real {
        self * self
    }
}

impl LapackScalar for num_complex::Complex<f32> {
    type Real = f32;
    fn conjugate(self) -> Self {
        self.conj()
    }
    fn abs_squared(self) -> Self::Real {
        self.norm_sqr()
    }
}

impl LapackScalar for num_complex::Complex<f64> {
    type Real = f64;
    fn conjugate(self) -> Self {
        self.conj()
    }
    fn abs_squared(self) -> Self::Real {
        self.norm_sqr()
    }
}
