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
