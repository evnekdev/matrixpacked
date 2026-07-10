mod common;
use common::assert_slice_close;
use matrixpacked::PackedSPD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::<f64>::from_vec(2, vec![4f64, 1f64, 3f64])?;
    let factor = a.cholesky_in_place()?;
    assert_eq!(factor.dimension(), 2);
    Ok(())
}
