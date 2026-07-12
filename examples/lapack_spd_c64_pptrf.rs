//! Demonstrates ZPPTRF, the LAPACK packed-storage routine.
//! Computes the Cholesky factorization of a positive-definite matrix in packed storage: `A = U^H*U` or `A = L*L^H`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedSPD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::<Complex64>::from_vec(2, vec![Complex64::new(4, 0), Complex64::new(1, 1), Complex64::new(3, 0)])?;
    let factor = a.cholesky_in_place()?;
    assert_eq!(factor.dimension(), 2);
    Ok(())
}
