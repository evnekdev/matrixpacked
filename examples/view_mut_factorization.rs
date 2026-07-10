use matrixpacked::{PackedSPDViewMut, PackedSymmetricViewMut};

fn main() {
    // No matrix allocation: LAPACK overwrites this borrowed packed buffer.
    let mut spd_data = [4.0_f64, 1.0, 1.0, 3.0, 0.0, 2.0];
    let spd = PackedSPDViewMut::from_slice_mut(3, &mut spd_data).unwrap();
    let cholesky = spd.cholesky_in_place().unwrap();
    let x = cholesky.solve_vector(&[9.0, 7.0, 7.0]).unwrap();
    assert!(x.iter().zip([1.0, 2.0, 3.0]).all(|(a,b)| (a-b).abs() < 1e-12));

    // Indefinite symmetric factorization reuses matrix storage and allocates only pivots.
    let mut sym_data = [0.0_f64, 1.0, 0.0, 2.0, 1.0, 3.0];
    let sym = PackedSymmetricViewMut::from_slice_mut(3, &mut sym_data).unwrap();
    let factor = sym.factorize_in_place().unwrap();
    let _ = factor.solve_vector(&[1.0, 2.0, 3.0]).unwrap();
}
