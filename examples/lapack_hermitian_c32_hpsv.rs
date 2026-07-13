//! Demonstrates CHPSV, the one-shot Hermitian-indefinite packed solve driver.
mod common;
use common::assert_slice_close;
use matrixpacked::PackedHermitian;
use num_complex::Complex32;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let a = PackedHermitian::from_vec(2, vec![c(0.0, 0.0), c(1.0, -1.0), c(0.0, 0.0)])?;
    let b = [c(2.0, 0.0), c(0.0, 1.0)];
    let expected = a.factorize()?.solve_vector(&b)?;
    assert_slice_close(&a.solve_once(&b, 1)?, &expected, 1e-5);
    Ok(())
}
