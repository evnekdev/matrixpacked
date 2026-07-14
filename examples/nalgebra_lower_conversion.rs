use matrixpacked::PackedLower;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Lower packed columns: [1, 2, 3], [4, 5], [6].
    let packed = PackedLower::from_vec(3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let matrix = packed.to_dmatrix()?;
    assert_eq!(
        matrix,
        DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0])
    );

    let source = DMatrix::from_row_slice(3, 3, &[1.0, 99.0, 98.0, 2.0, 4.0, 97.0, 3.0, 5.0, 6.0]);
    let extracted = PackedLower::from_lower_triangle(&source)?;
    assert_eq!(extracted.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}
