use matrixpacked::{PackedLower, RfpTranspose, Triangle};
use num_complex::Complex32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let packed = PackedLower::from_vec(
        3,
        vec![
            c(1., 1.),
            c(2., 2.),
            c(3., 3.),
            c(4., 4.),
            c(5., 5.),
            c(6., 6.),
        ],
    )?;
    let rfp = packed.to_rectangular_full_packed(RfpTranspose::Normal)?;
    assert_eq!(rfp.triangle(), Triangle::Lower);
    assert_eq!(rfp.shape(), (3, 2));
    assert_eq!(rfp.to_packed_lower()?.as_slice(), packed.as_slice());

    let transposed = packed.to_rectangular_full_packed(RfpTranspose::Transposed)?;
    assert_eq!(transposed.shape(), (2, 3));
    assert_eq!(transposed.to_packed_lower()?.as_slice(), packed.as_slice());
    Ok(())
}
