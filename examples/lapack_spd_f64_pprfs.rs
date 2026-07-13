use matrixpacked::PackedSPD;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::from_vec(2, vec![4.0f64, 1.0, 3.0])?;
    let f = a.cholesky()?;
    let b = [6.0, 7.0];
    let mut x = [0.9, 2.1];
    let r = f.refine_vector_in_place(&a, &b, &mut x)?;
    assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12);
    assert_eq!((r.forward_error.len(), r.backward_error.len()), (1, 1));
    Ok(())
}
