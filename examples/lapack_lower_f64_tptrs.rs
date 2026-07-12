//! Demonstrates DTPTRS, the LAPACK packed-storage routine.
//! Solves `A*X = B` (or a transpose/conjugate-transpose system) for a triangular matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedLower;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<f64>::from_vec(2, vec![2f64, 3f64, 4f64])?;
    let mut b = [2f64, 11f64];
    a.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[1f64, 2f64], 1e-10);
    Ok(())
}
