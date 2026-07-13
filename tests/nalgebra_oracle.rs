mod oracle;

use matrixpacked::{PackedHermitian, PackedLower, PackedSPD, PackedSymmetric, PackedUpper};
use nalgebra::DMatrix;
use num_complex::Complex64;
use oracle::{compare::*, convert::*, generate::*};

#[test]
fn lower_packing_and_direct_unpacking_round_trip() {
    let full = arbitrary_lower::<f64>(5, 0x10_01);
    let packed_data = pack_lower_column_major(&full);
    let packed = PackedLower::from_vec(5, packed_data.clone()).unwrap();

    assert_slice_close(
        &packed_data,
        packed.as_slice(),
        Tolerance::<f64>::for_scalar::<f64>(),
        5,
    );
    assert_matrix_close(
        &full,
        &decode_lower_packed(5, packed.as_slice()),
        Tolerance::<f64>::for_scalar::<f64>(),
    );
    assert_matrix_close(
        &full,
        &lower_to_dmatrix(&packed),
        Tolerance::<f64>::for_scalar::<f64>(),
    );
}

#[test]
fn upper_packing_and_direct_unpacking_round_trip() {
    let full = arbitrary_upper::<Complex64>(4, 0x10_02);
    let packed_data = pack_upper_column_major(&full);
    let packed = PackedUpper::from_vec(4, packed_data).unwrap();

    assert_matrix_close(
        &full,
        &decode_upper_packed(4, packed.as_slice()),
        Tolerance::<f64>::for_scalar::<Complex64>(),
    );
    assert_matrix_close(
        &full,
        &upper_to_dmatrix(&packed),
        Tolerance::<f64>::for_scalar::<Complex64>(),
    );
}

#[test]
fn symmetric_logical_expansion_mirrors_without_conjugation() {
    let full = complex_symmetric::<Complex64>(4, 0x10_03);
    let packed = PackedSymmetric::from_vec(4, pack_lower_column_major(&full)).unwrap();
    let actual = symmetric_to_dmatrix(&packed);

    assert_matrix_close(&full, &actual, Tolerance::<f64>::for_scalar::<Complex64>());
    assert_symmetric(&actual, Tolerance::<f64>::for_scalar::<Complex64>());
}

#[test]
fn hermitian_logical_expansion_conjugates_off_diagonal_entries() {
    let full = hermitian::<Complex64>(4, 0x10_04);
    let packed = PackedHermitian::from_vec(4, pack_lower_column_major(&full)).unwrap();
    let actual = hermitian_to_dmatrix(&packed);

    assert_matrix_close(&full, &actual, Tolerance::<f64>::for_scalar::<Complex64>());
    assert_hermitian(&actual, Tolerance::<f64>::for_scalar::<Complex64>());
}

#[test]
fn spd_generator_is_accepted_by_nalgebra_cholesky() {
    let full = spd_f64(7, 0x10_05, 0.5);
    assert!(full.clone().cholesky().is_some());

    let packed = PackedSPD::from_vec(7, pack_lower_column_major(&full)).unwrap();
    assert_matrix_close(
        &full,
        &spd_to_dmatrix(&packed),
        Tolerance::<f64>::for_scalar::<f64>(),
    );
}

#[test]
fn matrix_comparison_reports_a_deliberately_modified_entry() {
    let expected = DMatrix::<f64>::identity(3, 3);
    let mut actual = expected.clone();
    actual[(1, 2)] = 0.25;

    let failure = std::panic::catch_unwind(|| {
        assert_matrix_close(&expected, &actual, Tolerance::<f64>::for_scalar::<f64>());
    })
    .expect_err("the changed entry must be detected");
    let message = failure
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| failure.downcast_ref::<&str>().copied())
        .unwrap_or("");

    assert!(message.contains("dimension=3x3"), "{message}");
    assert!(message.contains("index=(1, 2)"), "{message}");
    assert!(message.contains("scalar=f64"), "{message}");
}

#[test]
fn deterministic_generators_repeat_for_the_same_seed() {
    let a = hpd_complex64(5, 0x10_06, 0.25);
    let b = hpd_complex64(5, 0x10_06, 0.25);
    assert_eq!(a, b);
}
