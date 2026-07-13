//! Demonstrates CHPTRD packed Hermitian tridiagonal reduction.
use matrixpacked::PackedHermitian;
use num_complex::Complex32;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let r = PackedHermitian::from_vec(
        3,
        vec![
            c(4., 0.),
            c(1., -1.),
            c(2., 0.5),
            c(3., 0.),
            c(0.5, -0.25),
            c(2., 0.),
        ],
    )?
    .tridiagonal_reduction()?;
    assert_eq!(r.diagonal().len(), 3);
    assert_eq!(r.off_diagonal().len(), 2);
    Ok(())
}
