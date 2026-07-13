use matrixpacked::{MatrixNorm, PackedSymmetric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::<f32>::from_vec(2, vec![-2.0, 0.0, 4.0])?;
    let anorm = a.matrix_norm(MatrixNorm::One)?;
    let rcond = a.factorize()?.rcond(anorm)?;
    assert!((rcond - 0.5).abs() <= 1.0e-5, "rcond={rcond}");
    Ok(())
}
