//! Demonstrates CSPTRS, the LAPACK packed-storage routine.
//! Solves `A*X = B` using the pivoted symmetric-indefinite factorization produced by `xSPTRF`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedSymmetricView;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = [Complex32::new(4.0, 0.0), Complex32::new(1.0, 0.0), Complex32::new(3.0, 0.0)];
    let a = PackedSymmetricView::<Complex32>::from_slice(2, &storage)?;
    let factor = a.factorize()?;
    let mut b = [Complex32::new(6.0, 0.0), Complex32::new(7.0, 0.0)];
    factor.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)], 1e-4);
    Ok(())
}
