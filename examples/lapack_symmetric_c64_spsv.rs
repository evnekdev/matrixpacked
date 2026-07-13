//! Demonstrates ZSPSV, the one-shot complex-symmetric packed solve driver.
mod common;
use common::assert_slice_close;
use matrixpacked::PackedSymmetric;
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let a = PackedSymmetric::from_vec(2, vec![c(0.0, 0.0), c(1.0, 1.0), c(0.0, 0.0)])?;
    let b = [c(2.0, 0.0), c(0.0, 1.0)];
    let expected = a.factorize()?.solve_vector(&b)?;
    assert_slice_close(&a.solve_once(&b, 1)?, &expected, 1e-12);
    Ok(())
}
