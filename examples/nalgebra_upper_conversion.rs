use matrixpacked::PackedUpper;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Upper packed columns: [1], [2, 4], [3, 5, 6].
    let packed = PackedUpper::from_vec(3, vec![1.0_f64, 2.0, 4.0, 3.0, 5.0, 6.0])?;
    let matrix = packed.to_dmatrix()?;
    assert_eq!(
        matrix,
        DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0])
    );

    let source = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 98.0, 97.0, 6.0]);
    let extracted = PackedUpper::from_upper_triangle(&source)?;
    assert_eq!(extracted.as_slice(), &[1.0, 2.0, 4.0, 3.0, 5.0, 6.0]);
    Ok(())
}
