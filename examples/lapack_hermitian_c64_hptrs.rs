//! Demonstrates ZHPTRS, the LAPACK packed-storage routine.
//! Solves `A*X = B` using the pivoted Hermitian-indefinite factorization produced by `xHPTRF`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex64;
use matrixpacked::PackedHermitianView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0), Complex64::new(-1.0, 0.0)];
    let a = PackedHermitianView::<Complex64>::from_slice(2, &storage)?;
    let factor = a.factorize()?;
    let mut b = [Complex64::new(4.0, -2.0), Complex64::new(-1.0, 1.0)];
    factor.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)], 1e-10);
    Ok(())
}
