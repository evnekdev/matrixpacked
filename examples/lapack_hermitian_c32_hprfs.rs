use matrixpacked::PackedHermitian;
use num_complex::Complex32;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex32::new;
    let a = PackedHermitian::from_vec(2, vec![c(4., 0.), c(1., 1.), c(3., 0.)])?;
    let f = a.factorize()?;
    let b = [c(5., -1.), c(4., 1.)];
    let mut x = [c(0.9, 0.), c(1.1, 0.)];
    let r = f.refine_vector_in_place(&a, &b, &mut x)?;
    assert!((x[0] - c(1., 0.)).norm() < 1e-4 && (x[1] - c(1., 0.)).norm() < 1e-4);
    assert_eq!((r.forward_error.len(), r.backward_error.len()), (1, 1));
    Ok(())
}
