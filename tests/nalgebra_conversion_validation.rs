#![cfg(feature = "nalgebra-interop")]

use matrixpacked::{
    ConversionTolerance, DefaultConversionTolerance, PackedHermitian, PackedLower,
    PackedMatrixError, PackedSPD, PackedSymmetric, PackedUpper,
};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};

fn exact() -> ConversionTolerance<f64> {
    ConversionTolerance::new(0.0, 0.0)
}

#[test]
fn exact_lower_and_upper_triangular_matrices_are_accepted() {
    let lower = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0]);
    let packed = PackedLower::try_from_dmatrix(&lower, exact()).unwrap();
    assert_eq!(packed.as_slice(), &[1.0, 2.0, 4.0, 3.0, 5.0, 6.0]);

    let upper = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 4.0, 0.0, 3.0, 5.0, 0.0, 0.0, 6.0]);
    let packed = PackedUpper::try_from_dmatrix(&upper, exact()).unwrap();
    assert_eq!(packed.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn triangular_noise_uses_tolerance_for_real_and_complex_values() {
    let tolerance = ConversionTolerance::new(1.0e-8, 0.0);
    let lower = DMatrix::from_row_slice(2, 2, &[1.0, 5.0e-9, 2.0, 3.0]);
    assert!(PackedLower::try_from_dmatrix(&lower, tolerance).is_ok());

    let c = Complex64::new;
    let upper = DMatrix::from_row_slice(
        2,
        2,
        &[c(1.0, 0.0), c(2.0, 3.0), c(3.0e-9, 4.0e-9), c(4.0, 0.0)],
    );
    assert!(PackedUpper::try_from_dmatrix(&upper, tolerance).is_ok());
}

#[test]
fn excessive_triangular_noise_reports_the_first_coordinate() {
    let lower = DMatrix::from_row_slice(2, 2, &[1.0, 2.0e-4, 2.0, 3.0]);
    assert_eq!(
        PackedLower::try_from_dmatrix(&lower, ConversionTolerance::new(1.0e-5, 0.0)).unwrap_err(),
        PackedMatrixError::NotTriangular {
            triangle: "lower",
            row: 0,
            column: 1,
        }
    );

    let upper = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0e-4, 4.0]);
    assert_eq!(
        PackedUpper::try_from_dmatrix(&upper, ConversionTolerance::new(1.0e-5, 0.0)).unwrap_err(),
        PackedMatrixError::NotTriangular {
            triangle: "upper",
            row: 1,
            column: 0,
        }
    );
}

#[test]
fn symmetric_validation_accepts_exact_and_tolerance_level_mismatches() {
    let exact_matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    assert!(PackedSymmetric::try_from_dmatrix(&exact_matrix, exact()).is_ok());

    let near = DMatrix::from_row_slice(2, 2, &[1.0, 2.0 + 1.0e-8, 2.0, 4.0]);
    let packed =
        PackedSymmetric::try_from_dmatrix(&near, ConversionTolerance::new(1.0e-8, 1.0e-8)).unwrap();
    assert_eq!(packed.as_slice(), &[1.0, 2.0, 4.0]);
}

#[test]
fn symmetric_validation_rejects_large_mismatches() {
    let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.1, 2.0, 4.0]);
    assert_eq!(
        PackedSymmetric::try_from_dmatrix(&matrix, ConversionTolerance::new(0.01, 0.0))
            .unwrap_err(),
        PackedMatrixError::NotSymmetric { row: 1, column: 0 }
    );
}

#[test]
fn complex_symmetric_validation_does_not_conjugate() {
    let c = Complex64::new;
    let symmetric =
        DMatrix::from_row_slice(2, 2, &[c(1.0, 1.0), c(2.0, 3.0), c(2.0, 3.0), c(4.0, 2.0)]);
    let packed = PackedSymmetric::try_from_dmatrix(&symmetric, exact()).unwrap();
    assert_eq!(packed.as_slice()[1], c(2.0, 3.0));

    let hermitian_pair =
        DMatrix::from_row_slice(2, 2, &[c(1.0, 0.0), c(2.0, -3.0), c(2.0, 3.0), c(4.0, 0.0)]);
    assert!(matches!(
        PackedSymmetric::try_from_dmatrix(&hermitian_pair, exact()),
        Err(PackedMatrixError::NotSymmetric { .. })
    ));
}

#[test]
fn hermitian_validation_accepts_exact_conjugates() {
    let c = Complex64::new;
    let matrix =
        DMatrix::from_row_slice(2, 2, &[c(3.0, 0.0), c(1.0, -2.0), c(1.0, 2.0), c(4.0, 0.0)]);
    let packed = PackedHermitian::try_from_dmatrix(&matrix, exact()).unwrap();
    assert_eq!(packed.as_slice(), &[c(3.0, 0.0), c(1.0, 2.0), c(4.0, 0.0)]);
}

#[test]
fn hermitian_validation_rejects_conjugate_mismatch() {
    let c = Complex32::new;
    let matrix =
        DMatrix::from_row_slice(2, 2, &[c(3.0, 0.0), c(1.0, -2.0), c(1.0, 2.2), c(4.0, 0.0)]);
    assert!(matches!(
        PackedHermitian::try_from_dmatrix(&matrix, ConversionTolerance::new(1.0e-4_f32, 0.0)),
        Err(PackedMatrixError::NotHermitian { row: 1, column: 0 })
    ));
}

#[test]
fn hermitian_diagonal_noise_is_validated_and_normalized() {
    let c = Complex64::new;
    let accepted = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
        c(2.0, 1.0e-9),
        c(3.0, -1.0e-9),
    ]));
    let packed =
        PackedHermitian::try_from_dmatrix(&accepted, ConversionTolerance::new(2.0e-9, 0.0))
            .unwrap();
    assert_eq!(packed.as_slice(), &[c(2.0, 0.0), c(0.0, 0.0), c(3.0, 0.0)]);

    let rejected = DMatrix::from_element(1, 1, c(2.0, 1.0e-3));
    assert_eq!(
        PackedHermitian::try_from_dmatrix(&rejected, ConversionTolerance::new(1.0e-6, 0.0))
            .unwrap_err(),
        PackedMatrixError::NonRealHermitianDiagonal { index: 0 }
    );
}

#[test]
fn valid_spd_and_hpd_matrices_are_accepted() {
    let spd = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    assert!(PackedSPD::try_from_dmatrix(&spd, exact()).is_ok());

    let c = Complex64::new;
    let hpd = DMatrix::from_row_slice(2, 2, &[c(4.0, 0.0), c(1.0, -1.0), c(1.0, 1.0), c(3.0, 0.0)]);
    assert!(PackedSPD::try_from_dmatrix(&hpd, exact()).is_ok());
}

#[test]
fn indefinite_and_semidefinite_matrices_are_rejected() {
    let indefinite = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    assert_eq!(
        PackedSPD::try_from_dmatrix(&indefinite, exact()).unwrap_err(),
        PackedMatrixError::NotPositiveDefinite
    );
    assert!(PackedSPD::try_from_structured_dmatrix(&indefinite, exact()).is_ok());

    let semidefinite = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(
        PackedSPD::try_from_dmatrix(&semidefinite, exact()).unwrap_err(),
        PackedMatrixError::NotPositiveDefinite
    );

    let c = Complex64::new;
    let hpd_indefinite =
        DMatrix::from_row_slice(2, 2, &[c(1.0, 0.0), c(2.0, -1.0), c(2.0, 1.0), c(1.0, 0.0)]);
    assert_eq!(
        PackedSPD::try_from_dmatrix(&hpd_indefinite, exact()).unwrap_err(),
        PackedMatrixError::NotPositiveDefinite
    );
}

#[test]
fn malformed_spd_structure_is_reported_before_cholesky() {
    let malformed = DMatrix::from_row_slice(2, 2, &[-1.0, 3.0, 2.0, -1.0]);
    assert_eq!(
        PackedSPD::try_from_dmatrix(&malformed, exact()).unwrap_err(),
        PackedMatrixError::NotSymmetric { row: 1, column: 0 }
    );
}

#[test]
fn badly_scaled_positive_definite_matrix_is_accepted() {
    let matrix = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0e-200, 1.0e200]));
    assert!(PackedSPD::try_from_dmatrix(&matrix, exact()).is_ok());
}

#[test]
fn tolerance_validation_and_absolute_relative_behavior() {
    let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 1000.001, 1000.0, 2.0]);
    assert!(PackedSymmetric::try_from_dmatrix(&matrix, exact()).is_err());
    assert!(
        PackedSymmetric::try_from_dmatrix(&matrix, ConversionTolerance::new(0.002, 0.0)).is_ok()
    );
    assert!(
        PackedSymmetric::try_from_dmatrix(&matrix, ConversionTolerance::new(0.0, 2.0e-6)).is_ok()
    );

    let negative = ConversionTolerance::new(-1.0, 0.0);
    assert!(matches!(
        PackedSymmetric::try_from_dmatrix(&matrix, negative),
        Err(PackedMatrixError::InvalidTolerance {
            component: "absolute",
            ..
        })
    ));
    let nan = ConversionTolerance::new(0.0, f64::NAN);
    assert!(matches!(
        PackedSymmetric::try_from_dmatrix(&matrix, nan),
        Err(PackedMatrixError::InvalidTolerance {
            component: "relative",
            ..
        })
    ));

    let default = f64::default_conversion_tolerance();
    assert_eq!(default, ConversionTolerance::<f64>::default());
    assert_eq!(default.absolute, 0.0);
    assert!(default.relative > 0.0);
}

#[test]
fn strict_conversions_reject_non_square_matrices() {
    let matrix = DMatrix::from_element(2, 3, 0.0_f64);
    let expected = PackedMatrixError::NonSquareMatrix {
        rows: 2,
        columns: 3,
    };
    assert_eq!(
        PackedLower::try_from_dmatrix(&matrix, exact()).unwrap_err(),
        expected
    );
    assert_eq!(
        PackedUpper::try_from_dmatrix(&matrix, exact()).unwrap_err(),
        expected
    );
    assert_eq!(
        PackedSymmetric::try_from_dmatrix(&matrix, exact()).unwrap_err(),
        expected
    );
    assert_eq!(
        PackedHermitian::try_from_dmatrix(&matrix, exact()).unwrap_err(),
        expected
    );
    assert_eq!(
        PackedSPD::try_from_dmatrix(&matrix, exact()).unwrap_err(),
        expected
    );
}
