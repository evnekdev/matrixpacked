//! Demonstrates CSPTRF, the LAPACK packed-storage routine.
//! Computes a pivoted diagonal factorization of a symmetric indefinite packed matrix: `A = U*D*U^T` or `A = L*D*L^T`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedSymmetric;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::<Complex32>::from_vec(2, vec![Complex32::new(4.0, 0.0), Complex32::new(1.0, 0.0), Complex32::new(3.0, 0.0)])?;
    let factor = a.factorize_in_place()?;
    assert_eq!(factor.dimension(), 2);
    assert_eq!(factor.pivots().len(), 2);
    Ok(())
}
