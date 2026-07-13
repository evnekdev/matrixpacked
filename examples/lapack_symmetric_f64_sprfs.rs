use matrixpacked::PackedSymmetric;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSymmetric::from_vec(2, vec![2.0f64, 1.0, -1.0])?;
    let f = a.factorize()?;
    let b = [4.0, -1.0];
    let mut x = [0.8, 2.2];
    let r = f.refine_vector_in_place(&a, &b, &mut x)?;
    assert!((x[0] - 1.).abs() < 1e-12 && (x[1] - 2.).abs() < 1e-12);
    assert_eq!((r.forward_error.len(), r.backward_error.len()), (1, 1));
    Ok(())
}
