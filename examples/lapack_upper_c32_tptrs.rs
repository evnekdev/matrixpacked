//! Demonstrates CTPTRS, the LAPACK packed-storage routine.
//! Solves `A*X = B` (or a transpose/conjugate-transpose system) for a triangular matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use num_complex::Complex32;
use matrixpacked::PackedUpper;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<Complex32>::from_vec(2, vec![Complex32::new(2, 0), Complex32::new(3, 0), Complex32::new(4, 0)])?;
    let mut b = [Complex32::new(8, 0), Complex32::new(8, 0)];
    a.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[Complex32::new(1, 0), Complex32::new(2, 0)], 1e-4);
    Ok(())
}
