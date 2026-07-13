//! Demonstrates ZTPTRS, the LAPACK packed-storage routine.
//! Solves `A*X = B` (or a transpose/conjugate-transpose system) for a triangular matrix `A` stored in packed form.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedUpper;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<Complex64>::from_vec(
        2,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )?;
    let mut b = [Complex64::new(8.0, 0.0), Complex64::new(8.0, 0.0)];
    a.solve_vector_in_place(&mut b)?;
    assert_slice_close(
        &b,
        &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        1e-10,
    );
    Ok(())
}
