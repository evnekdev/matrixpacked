mod common;
use common::assert_slice_close;
use matrixpacked::PackedUpper;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedUpper::<f64>::from_vec(2, vec![2f64, 3f64, 4f64])?;
    let mut b = [8f64, 8f64];
    a.solve_vector_in_place(&mut b)?;
    assert_slice_close(&b, &[1f64, 2f64], 1e-10);
    Ok(())
}
