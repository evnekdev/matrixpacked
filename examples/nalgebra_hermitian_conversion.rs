use matrixpacked::PackedHermitian;
use nalgebra::DMatrix;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let packed = PackedHermitian::from_vec(2, vec![c(1.0, 8.0), c(2.0, 3.0), c(4.0, 9.0)])?;
    let matrix = packed.to_dmatrix();
    assert_eq!(matrix[(0, 0)], c(1.0, 0.0));
    assert_eq!(matrix[(0, 1)], c(2.0, -3.0));
    assert_eq!(matrix[(1, 0)], c(2.0, 3.0));

    let extracted = PackedHermitian::from_lower_triangle(&DMatrix::from_row_slice(
        2,
        2,
        &[c(1.0, 7.0), c(99.0, 0.0), c(2.0, 3.0), c(4.0, 6.0)],
    ))?;
    assert_eq!(
        extracted.as_slice(),
        &[c(1.0, 0.0), c(2.0, 3.0), c(4.0, 0.0)]
    );
    Ok(())
}
