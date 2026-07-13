use matrixpacked::{MatrixNorm, PackedSPD};
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::<Complex32>::from_vec(
        2,
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(4.0, 0.0),
        ],
    )?;
    let anorm = a.matrix_norm(MatrixNorm::One)?;
    let rcond = a.cholesky()?.rcond(anorm)?;
    assert!((rcond - 0.5).abs() <= 1.0e-5, "rcond={rcond}");
    Ok(())
}
