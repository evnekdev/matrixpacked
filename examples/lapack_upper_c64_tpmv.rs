//! Demonstrates ZTPMV, the BLAS packed-storage routine.
//! Computes `x := A*x` (or a transpose/conjugate-transpose variant) for a triangular matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedUpperView;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let a = PackedUpperView::<Complex64>::from_slice(2, &storage)?;
    let x = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let y = a.mul_vector(&x)?;
    assert_slice_close(
        &y,
        &[Complex64::new(8.0, 0.0), Complex64::new(8.0, 0.0)],
        1e-10,
    );
    Ok(())
}
