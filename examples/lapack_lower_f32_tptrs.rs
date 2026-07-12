//! Demonstrates STPTRS, the LAPACK packed-storage routine.
//! Solves `A*X = B` (or a transpose/conjugate-transpose system) for a triangular matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedLower;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedLower::<f32>::from_vec(2, vec![2f32, 3f32, 4f32])?;
    let mut b = [2f32, 11f32];
    a.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[1f32, 2f32], 1e-4);
    Ok(())
}
