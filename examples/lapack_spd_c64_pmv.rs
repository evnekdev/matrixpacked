//! Demonstrates ZHPMV, the BLAS packed-storage routine.
//! Computes `y := alpha*A*x + beta*y` for a complex Hermitian matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedSPDView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [Complex64::new(4.0, 0.0), Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)];
    let a = PackedSPDView::<Complex64>::from_slice(2, &storage)?;
    let x = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let y = a.mul_vector(&x)?;
    assert_slice_close(&y, &[Complex64::new(6.0, -2.0), Complex64::new(7.0, 1.0)], 1e-10);
    Ok(())
}
