use matrixpacked::{FullTriangular, PackedUpper, Triangle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let full = FullTriangular::from_vec(
        3,
        Triangle::Upper,
        vec![1.0_f64, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0],
    )?;
    let packed = PackedUpper::from_full_triangular(&full)?;
    assert_eq!(packed.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(packed.to_full_triangular()?.as_slice(), full.as_slice());
    Ok(())
}
