//! Demonstrates ZTPMV, the BLAS packed-storage routine.
//! Computes `x := A*x` (or a transpose/conjugate-transpose variant) for a triangular matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::{PackedLower, PackedLowerView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [Complex64::new(2, 0), Complex64::new(3, 0), Complex64::new(4, 0)];
    let a = PackedLowerView::<Complex64>::from_slice(2, &storage)?;
    let x = [Complex64::new(1, 0), Complex64::new(2, 0)];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[Complex64::new(2, 0), Complex64::new(11, 0)], 1e-10);
    Ok(())
}
