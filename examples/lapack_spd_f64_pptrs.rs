//! Demonstrates DPPTRS, the LAPACK packed-storage routine.
//! Solves `A*X = B` from the packed Cholesky factorization produced by `xPPTRF`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPDView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [4f64, 1f64, 3f64];
    let a = PackedSPDView::<f64>::from_slice(2, &storage)?;
    let factor = a.cholesky()?;
    let mut b = [6f64, 7f64];
    factor.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[1f64, 2f64], 1e-10);
    Ok(())
}
