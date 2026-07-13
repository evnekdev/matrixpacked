use matrixpacked::{PackedUpper, RectangularFullPackedView, RfpTranspose, Triangle};
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let packed = PackedUpper::from_vec(
        4,
        vec![
            c(1., 1.),
            c(2., 2.),
            c(3., 3.),
            c(4., 4.),
            c(5., 5.),
            c(6., 6.),
            c(7., 7.),
            c(8., 8.),
            c(9., 9.),
            c(10., 10.),
        ],
    )?;
    let rfp = packed.to_rectangular_full_packed(RfpTranspose::Transposed)?;
    assert_eq!(rfp.triangle(), Triangle::Upper);
    assert_eq!(rfp.shape(), (2, 5));

    let view = RectangularFullPackedView::from_slice(
        4,
        Triangle::Upper,
        RfpTranspose::Transposed,
        rfp.as_slice(),
    )?;
    assert_eq!(view.to_packed_upper()?.as_slice(), packed.as_slice());
    Ok(())
}
