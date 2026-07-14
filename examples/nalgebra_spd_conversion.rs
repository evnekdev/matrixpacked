use matrixpacked::PackedSPD;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let packed = PackedSPD::from_vec(2, vec![4.0_f64, 1.0, 3.0])?;
    assert_eq!(
        packed.to_dmatrix(),
        DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0])
    );

    // Extraction names its unchecked numerical semantics: this does not prove SPD.
    let extracted = PackedSPD::from_lower_triangle_unchecked_structure(&DMatrix::from_row_slice(
        2,
        2,
        &[1.0, 99.0, 2.0, -3.0],
    ))?;
    assert_eq!(extracted.as_slice(), &[1.0, 2.0, -3.0]);
    Ok(())
}
