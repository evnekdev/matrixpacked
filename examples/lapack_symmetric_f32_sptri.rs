//! Demonstrates SSPTRI, the LAPACK packed-storage routine.
//! Computes the inverse of a symmetric indefinite packed matrix from the factorization produced by `xSPTRF`.
//! Here `x` in a routine family name stands for the scalar type (`S`, `D`, `C`, or `Z`).

mod common;
use common::assert_slice_close;
use matrixpacked::PackedSymmetricViewMut;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [4f32, 1f32, 3f32];
    let a = PackedSymmetricViewMut::<f32>::from_slice_mut(2, &mut storage)?;
    let mut factor = a.factorize_in_place()?;
    factor.inverse_in_place()?;
    assert_slice_close(
        factor.as_slice(),
        &[0.272_727_28_f32, -0.090_909_09_f32, 0.363_636_37_f32],
        1e-4,
    );
    Ok(())
}
