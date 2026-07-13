use matrixpacked::{
    Diagonal, PackedHermitian, PackedLower, PackedLowerViewMut, PackedMatrixError, PackedSPD,
    PackedSymmetric, PackedUpper,
};
use num_complex::{Complex32, Complex64};

fn close(a: &[f64], b: &[f64]) {
    assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-11));
}

#[test]
fn triangular_owned_allocating_unit_and_singular() {
    let lower = PackedLower::from_vec(2, vec![2.0_f64, 3.0, 4.0]).unwrap();
    let inverse = lower.inverse().unwrap();
    let x = [1.5, -2.0];
    close(
        &inverse.mul_vector(&lower.mul_vector(&x).unwrap()).unwrap(),
        &x,
    );

    let upper = PackedUpper::from_vec(2, vec![2.0_f64, 3.0, 4.0]).unwrap();
    let inverse = upper.into_inverse().unwrap();
    close(&inverse.mul_vector(&[2.0, 4.0]).unwrap(), &[-0.5, 1.0]);

    let unit = PackedLower::from_vec(2, vec![99.0_f64, 3.0, 99.0]).unwrap();
    let inverse = unit.inverse_with_diagonal(Diagonal::Unit).unwrap();
    // LAPACK does not reference or overwrite stored diagonal entries in unit mode.
    close(inverse.as_slice(), &[99.0, -3.0, 99.0]);

    let singular = PackedUpper::from_vec(2, vec![0.0_f64, 1.0, 2.0]).unwrap();
    assert!(matches!(
        singular.inverse(),
        Err(PackedMatrixError::FactorizationFailure { .. })
    ));
}

#[test]
fn triangular_mutable_view_and_complex() {
    let mut data = [
        Complex32::new(2.0, 0.0),
        Complex32::new(1.0, 1.0),
        Complex32::new(3.0, 0.0),
    ];
    let mut view = PackedLowerViewMut::from_slice_mut(2, &mut data).unwrap();
    view.inverse_in_place().unwrap();
    assert!((view.as_slice()[0] - Complex32::new(0.5, 0.0)).norm() < 1e-6);
}

#[test]
fn structured_allocating_inverses_recover_vectors() {
    let spd = PackedSPD::from_vec(2, vec![4.0_f64, 1.0, 3.0]).unwrap();
    let spd_inverse = spd.inverse().unwrap();
    let x = [2.0, -1.0];
    close(
        &spd_inverse
            .mul_vector(&spd.mul_vector(&x).unwrap())
            .unwrap(),
        &x,
    );

    let symmetric = PackedSymmetric::from_vec(2, vec![0.0_f64, 1.0, 0.0]).unwrap();
    let symmetric_inverse = symmetric.inverse().unwrap();
    close(
        &symmetric_inverse
            .mul_vector(&symmetric.mul_vector(&x).unwrap())
            .unwrap(),
        &x,
    );

    let hermitian = PackedHermitian::from_vec(
        2,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    let inverse = hermitian.inverse().unwrap();
    let x = [Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.5)];
    let recovered = inverse
        .mul_vector(&hermitian.mul_vector(&x).unwrap())
        .unwrap();
    assert!(
        recovered
            .iter()
            .zip(x)
            .all(|(a, b)| (*a - b).norm() < 1e-11)
    );
}

#[test]
fn empty_and_one_by_one_are_consistent() {
    assert!(
        PackedLower::<f32>::from_vec(0, vec![])
            .unwrap()
            .inverse()
            .unwrap()
            .as_slice()
            .is_empty()
    );
    let one = PackedSPD::from_vec(1, vec![4.0_f32])
        .unwrap()
        .inverse()
        .unwrap();
    assert!((one.as_slice()[0] - 0.25).abs() < 1e-6);
}
