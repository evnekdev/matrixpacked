//! Checks the BLAS `ZHPR2` packed Hermitian rank-2 update.
use matrixpacked::PackedHermitian;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let x = [c(1.0, 1.0), c(-2.0, 0.5)];
    let y = [c(0.5, -1.0), c(1.0, 2.0)];
    let alpha = c(0.25, -0.5);
    let mut a = PackedHermitian::from_vec(2, vec![c(2.0, 0.0), c(1.0, -1.0), c(3.0, 0.0)])?;
    let original = (0..2)
        .map(|i| (0..2).map(|j| a.get(i, j).unwrap()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    a.rank2_update_in_place(alpha, &x, &y)?;
    for i in 0..2 {
        for j in 0..2 {
            let expected =
                original[i][j] + alpha * x[i] * y[j].conj() + alpha.conj() * y[i] * x[j].conj();
            assert!((a.get(i, j)? - expected).norm() < 1e-12);
        }
        assert_eq!(a.get(i, i)?.im, 0.0);
    }
    assert!((a.get(0, 1)? - a.get(1, 0)?.conj()).norm() < 1e-12);
    Ok(())
}
