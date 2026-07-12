//! Demonstrates SSPTRF, the LAPACK packed-storage routine.
//! Computes a pivoted diagonal factorization of a symmetric indefinite packed matrix: `A = U*D*U^T` or `A = L*D*L^T`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedSymmetric;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::<f32>::from_vec(2, vec![4f32, 1f32, 3f32])?;
    let factor = a.factorize_in_place()?;
    assert_eq!(factor.dimension(), 2);
    assert_eq!(factor.pivots().len(), 2);
    Ok(())
}
