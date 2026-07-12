//! Demonstrates CTPMV, the BLAS packed-storage routine.
//! Computes `x := A*x` (or a transpose/conjugate-transpose variant) for a triangular matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::{PackedLower, PackedLowerView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [Complex32::new(2.0, 0.0), Complex32::new(3.0, 0.0), Complex32::new(4.0, 0.0)];
    let a = PackedLowerView::<Complex32>::from_slice(2, &storage)?;
    let x = [Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[Complex32::new(2.0, 0.0), Complex32::new(11.0, 0.0)], 1e-4);
    Ok(())
}
