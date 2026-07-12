//! Demonstrates DPPTRF, the LAPACK packed-storage routine.
//! Computes the Cholesky factorization of a positive-definite matrix in packed storage: `A = U^H*U` or `A = L*L^H`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::<f64>::from_vec(2, vec![4f64, 1f64, 3f64])?;
    let factor = a.cholesky_in_place()?;
    assert_eq!(factor.dimension(), 2);
    Ok(())
}
