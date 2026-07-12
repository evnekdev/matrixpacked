//! Demonstrates ZPPTRS, the LAPACK packed-storage routine.
//! Solves `A*X = B` from the packed Cholesky factorization produced by `xPPTRF`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedSPDView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [Complex64::new(4, 0), Complex64::new(1, 1), Complex64::new(3, 0)];
    let a = PackedSPDView::<Complex64>::from_slice(2, &storage)?;
    let factor = a.cholesky()?;
    let mut b = [Complex64::new(6, -2), Complex64::new(7, 1)];
    factor.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[Complex64::new(1, 0), Complex64::new(2, 0)], 1e-10);
    Ok(())
}
