//! Demonstrates CHPTRI, the LAPACK packed-storage routine.
//! Computes the inverse of a Hermitian indefinite packed matrix from the factorization produced by `xHPTRF`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedHermitianViewMut;
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [
        Complex32::new(2.0, 0.0),
        Complex32::new(1.0, 1.0),
        Complex32::new(-1.0, 0.0),
    ];
    let a = PackedHermitianViewMut::<Complex32>::from_slice_mut(2, &mut storage)?;
    let mut factor = a.factorize_in_place()?;
    factor.inverse_in_place()?;
    assert_slice_close(
        factor.as_slice(),
        &[
            Complex32::new(0.25, 0.0),
            Complex32::new(0.25, 0.25),
            Complex32::new(-0.5, 0.0),
        ],
        1e-4,
    );
    Ok(())
}
