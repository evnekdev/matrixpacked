//! Demonstrates CHPTRF, the LAPACK packed-storage routine.
//! Computes a pivoted diagonal factorization of a Hermitian indefinite packed matrix: `A = U*D*U^H` or `A = L*D*L^H`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedHermitian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedHermitian::<Complex32>::from_vec(2, vec![Complex32::new(2.0, 0.0), Complex32::new(1.0, 1.0), Complex32::new(-1.0, 0.0)])?;
    let factor = a.factorize_in_place()?;
    assert_eq!(factor.dimension(), 2);
    Ok(())
}
