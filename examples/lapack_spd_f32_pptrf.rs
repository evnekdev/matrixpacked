//! Demonstrates SPPTRF, the LAPACK packed-storage routine.
//! Computes the Cholesky factorization of a positive-definite matrix in packed storage: `A = U^H*U` or `A = L*L^H`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use matrixpacked::PackedSPD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::<f32>::from_vec(2, vec![4f32, 1f32, 3f32])?;
    let factor = a.cholesky_in_place()?;
    assert_eq!(factor.dimension(), 2);
    Ok(())
}
