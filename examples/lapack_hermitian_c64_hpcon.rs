use matrixpacked::{MatrixNorm, PackedHermitian};
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedHermitian::<Complex64>::from_vec(
        2,
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )?;
    let anorm = a.matrix_norm(MatrixNorm::One)?;
    let rcond = a.factorize()?.rcond(anorm)?;
    assert!((rcond - 0.5).abs() <= 1.0e-12, "rcond={rcond}");
    Ok(())
}
