#![allow(dead_code)]

use num_complex::{Complex32, Complex64};

pub trait Approx {
    fn approx_eq(self, rhs: Self, tol: f64) -> bool;
}
impl Approx for f32 {
    fn approx_eq(self, rhs: Self, tol: f64) -> bool {
        (self - rhs).abs() <= tol as f32
    }
}
impl Approx for f64 {
    fn approx_eq(self, rhs: Self, tol: f64) -> bool {
        (self - rhs).abs() <= tol
    }
}
impl Approx for Complex32 {
    fn approx_eq(self, rhs: Self, tol: f64) -> bool {
        (self - rhs).norm() <= tol as f32
    }
}
impl Approx for Complex64 {
    fn approx_eq(self, rhs: Self, tol: f64) -> bool {
        (self - rhs).norm() <= tol
    }
}

pub fn assert_slice_close<T: Approx + Copy + core::fmt::Debug>(
    actual: &[T],
    expected: &[T],
    tol: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.approx_eq(e, tol),
            "index {i}: actual={a:?}, expected={e:?}"
        );
    }
}
