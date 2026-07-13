//! Demonstrates CHPMV, the BLAS packed-storage routine.
//! Computes `y := alpha*A*x + beta*y` for a complex Hermitian matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedHermitianView;
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [
        Complex32::new(2.0, 0.0),
        Complex32::new(1.0, 1.0),
        Complex32::new(-1.0, 0.0),
    ];
    let a = PackedHermitianView::<Complex32>::from_slice(2, &storage)?;
    let y = a.mul_vector(&[Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)])?;
    assert_slice_close(
        &y,
        &[Complex32::new(4.0, -2.0), Complex32::new(-1.0, 1.0)],
        1e-4,
    );
    Ok(())
}
