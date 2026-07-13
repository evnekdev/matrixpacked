//! Checks the BLAS `SSPR` packed symmetric rank-1 update.
use matrixpacked::PackedSymmetric;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = [1.0f32, -2.0, 0.5];
    let alpha = 0.75;
    let mut a = PackedSymmetric::from_fn(3, |i, j| (i + j + 1) as f32)?;
    let original = (0..3)
        .map(|i| (0..3).map(|j| a.get(i, j).unwrap()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    a.rank1_update_in_place(alpha, &x)?;
    for i in 0..3 {
        for j in 0..3 {
            let expected = original[i][j] + alpha * x[i] * x[j];
            assert!((a.get(i, j)? - expected).abs() < 1e-5);
        }
    }
    Ok(())
}
