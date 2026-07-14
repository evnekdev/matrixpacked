use matrixpacked::{GeneralizedEigenproblem, PackedSPD, PackedSymmetric};
use nalgebra::{DMatrix, DVector};
use proptest::{
    prelude::*,
    test_runner::{Config as ProptestConfig, RngSeed},
};

use super::{
    compare::{Tolerance, assert_identity_close, assert_matrix_close, matrix_rhs_residual},
    convert::{pack_lower_column_major, spd_to_dmatrix, symmetric_to_dmatrix},
    generate::{
        deliberately_ill_conditioned_spd_f64, moderately_conditioned_spd_f64, real_symmetric_f64,
        singular_psd_f64, vector, well_conditioned_spd_f64,
    },
};

const FAST_CASES: u32 = 16;
const FIXED_RUNNER_SEED: u64 = 0x4d50_4143_4b45_4450;

/// Shared proptest profile. Ordinary `cargo test` is intentionally fast and
/// deterministic; CI can raise the case count without changing test code.
pub fn property_config() -> ProptestConfig {
    let mut config = ProptestConfig::default();
    config.cases = std::env::var("MATRIXPACKED_PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(FAST_CASES);
    config.rng_seed = RngSeed::Fixed(
        std::env::var("MATRIXPACKED_PROPTEST_SEED")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(FIXED_RUNNER_SEED),
    );
    config
}

pub fn matrix_dimension() -> impl Strategy<Value = usize> {
    0usize..=24
}
pub fn nonsingular_dimension() -> impl Strategy<Value = usize> {
    1usize..=24
}
pub fn indefinite_dimension() -> impl Strategy<Value = usize> {
    2usize..=24
}
pub fn deterministic_seed() -> impl Strategy<Value = u64> {
    any::<u64>()
}
pub fn rhs_count() -> impl Strategy<Value = usize> {
    1usize..=8
}

fn diagnostics(
    family: &str,
    n: usize,
    seed: u64,
    packed: &[f64],
    full: &DMatrix<f64>,
    options: &str,
) -> String {
    format!(
        "scalar=f64 family={family} dimension={n} seed={seed:#x} options={options} packed={packed:?} full={full:?}"
    )
}

fn generalized_problem(index: u8) -> GeneralizedEigenproblem {
    match index % 3 {
        0 => GeneralizedEigenproblem::AxEqualsLambdaBx,
        1 => GeneralizedEigenproblem::ABxEqualsLambdaX,
        _ => GeneralizedEigenproblem::BAxEqualsLambdaX,
    }
}

proptest! {
    #![proptest_config(property_config())]

    #[test]
    fn property_spd_multi_rhs_inverse_and_refinement(
        n in 1usize..=10,
        nrhs in 1usize..=4,
        seed in any::<u64>(),
    ) {
        let a = well_conditioned_spd_f64(n, seed);
        let packed_data = pack_lower_column_major(&a);
        let matrix = PackedSPD::from_vec(n, packed_data.clone()).unwrap();
        let factor = matrix.cholesky().unwrap();
        let exact = DMatrix::from_fn(n, nrhs, |row, col| {
            ((seed as f64 * 1e-7) + row as f64 * 0.31 - col as f64 * 0.17).sin()
        });
        let rhs = &a * &exact;
        let mut solved = rhs.as_slice().to_vec();
        factor.solve_many_in_place(&mut solved, nrhs).unwrap();
        let solved_matrix = DMatrix::from_column_slice(n, nrhs, &solved);
        let residual = matrix_rhs_residual(&a, &solved_matrix, &rhs);
        let context = diagnostics("well-conditioned SPD", n, seed, &packed_data, &a, &format!("nrhs={nrhs}"));
        prop_assert!(residual < 2.0e-12, "{context} residual={residual:e}");

        for column in 0..nrhs {
            let separately = factor.solve_vector(rhs.column(column).as_slice()).unwrap();
            let difference = (&DVector::from_vec(separately) - solved_matrix.column(column)).norm();
            prop_assert!(difference < 2.0e-11, "{context} column={column} difference={difference:e}");
        }

        let inverse = spd_to_dmatrix(&matrix.inverse().unwrap());
        let identity = DMatrix::identity(n, n);
        let left = matrix_rhs_residual(&a, &inverse, &identity);
        let right = matrix_rhs_residual(&inverse, &a, &identity);
        prop_assert!(left < 2.0e-12 && right < 2.0e-12, "{context} left_inverse_residual={left:e} right_inverse_residual={right:e}");

        let mut perturbed = exact.clone();
        for (index, value) in perturbed.as_mut_slice().iter_mut().enumerate() {
            *value += (index + 1) as f64 * 1.0e-8;
        }
        let before = matrix_rhs_residual(&a, &perturbed, &rhs);
        let report = factor
            .refine_many_in_place(&matrix, rhs.as_slice(), perturbed.as_mut_slice(), nrhs)
            .unwrap();
        let after = matrix_rhs_residual(&a, &perturbed, &rhs);
        prop_assert!(after <= before * 1.001 + 1.0e-15, "{context} refinement_before={before:e} refinement_after={after:e} report={report:?}");
    }

    #[test]
    fn property_symmetric_rank_updates_match_dense(
        n in 1usize..=12,
        seed in any::<u64>(),
        alpha in -1.0f64..1.0,
    ) {
        let a = real_symmetric_f64(n, seed);
        let x = vector::<f64>(n, seed ^ 0x81_1000);
        let y = vector::<f64>(n, seed ^ 0x81_2000);
        let packed_data = pack_lower_column_major(&a);
        let context = diagnostics("symmetric", n, seed, &packed_data, &a, &format!("rank updates alpha={alpha}"));

        let mut rank1 = PackedSymmetric::from_vec(n, packed_data.clone()).unwrap();
        rank1.rank1_update_in_place(alpha, x.as_slice()).unwrap();
        let expected_rank1 = &a + alpha * (&x * x.transpose());
        let error1 = (symmetric_to_dmatrix(&rank1) - &expected_rank1).norm();
        prop_assert!(error1 <= 1.0e-11 * expected_rank1.norm().max(1.0), "{context} rank1_difference={error1:e}");

        let mut rank2 = PackedSymmetric::from_vec(n, packed_data).unwrap();
        rank2.rank2_update_in_place(alpha, x.as_slice(), y.as_slice()).unwrap();
        let expected_rank2 = &a + alpha * (&x * y.transpose() + &y * x.transpose());
        let error2 = (symmetric_to_dmatrix(&rank2) - &expected_rank2).norm();
        prop_assert!(error2 <= 2.0e-11 * expected_rank2.norm().max(1.0), "{context} rank2_difference={error2:e}");
    }

    #[test]
    fn property_generalized_eigenpairs_match_controlled_standard_problem(
        n in 1usize..=10,
        seed in any::<u64>(),
        problem_index in 0u8..3,
    ) {
        let problem = generalized_problem(problem_index);
        let q = DMatrix::from_fn(n, n, |row, col| {
            (((row + 1) * (col + 2)) as f64 + seed as f64 * 1e-7).sin()
                + if row == col { 1.5 } else { 0.0 }
        }).qr().q();
        let expected: Vec<f64> = (0..n).map(|index| -2.0 + index as f64 * 0.75).collect();
        let standard = &q * DMatrix::from_diagonal(&DVector::from_vec(expected.clone())) * q.transpose();
        let l: Vec<f64> = (0..n).map(|index| 1.0 + index as f64 * 0.07).collect();
        let b = DMatrix::from_diagonal(&DVector::from_iterator(n, l.iter().map(|value| value * value)));
        let a = DMatrix::from_fn(n, n, |row, col| match problem {
            GeneralizedEigenproblem::AxEqualsLambdaBx => standard[(row, col)] * l[row] * l[col],
            GeneralizedEigenproblem::ABxEqualsLambdaX | GeneralizedEigenproblem::BAxEqualsLambdaX => {
                standard[(row, col)] / (l[row] * l[col])
            }
        });
        let a_data = pack_lower_column_major(&a);
        let b_data = pack_lower_column_major(&b);
        let packed_a = PackedSymmetric::from_vec(n, a_data.clone()).unwrap();
        let packed_b = PackedSPD::from_vec(n, b_data).unwrap();
        let result = packed_a.generalized_eigendecomposition_divide_conquer(&packed_b, problem).unwrap();
        let vectors = DMatrix::from_column_slice(n, n, result.eigenvectors.as_ref().unwrap());
        let context = diagnostics("generalized symmetric-definite", n, seed, &a_data, &a, &format!("problem={problem:?}"));

        for (index, (&actual, &known)) in result.eigenvalues.iter().zip(&expected).enumerate() {
            let difference = (actual - known).abs();
            prop_assert!(difference <= 2.0e-10 * actual.abs().max(known.abs()).max(1.0), "{context} eigenvalue_index={index} actual={actual:e} expected={known:e} difference={difference:e}");
            let v = vectors.column(index).into_owned();
            let (left, right) = match problem {
                GeneralizedEigenproblem::AxEqualsLambdaBx => (&a * &v, (&b * &v) * actual),
                GeneralizedEigenproblem::ABxEqualsLambdaX => (&a * (&b * &v), v * actual),
                GeneralizedEigenproblem::BAxEqualsLambdaX => (&b * (&a * &v), v * actual),
            };
            let residual = (&left - &right).norm()
                / ((a.norm() * b.norm() + actual.abs()) * left.norm().max(1.0));
            prop_assert!(residual < 2.0e-10, "{context} eigenvalue_index={index} generalized_residual={residual:e}");
        }

        let metric_vectors = match problem {
            GeneralizedEigenproblem::AxEqualsLambdaBx | GeneralizedEigenproblem::ABxEqualsLambdaX => &b * &vectors,
            GeneralizedEigenproblem::BAxEqualsLambdaX => b.clone().cholesky().unwrap().solve(&vectors),
        };
        let normalization_error = (vectors.transpose() * metric_vectors - DMatrix::identity(n, n)).norm();
        prop_assert!(normalization_error < 2.0e-10, "{context} normalization_error={normalization_error:e}");
    }
}

#[test]
fn extended_property_numerical_stress() {
    if std::env::var("MATRIXPACKED_EXTENDED_TESTS").as_deref() != Ok("1") {
        return;
    }

    for (family, generate) in [
        (
            "well-conditioned",
            well_conditioned_spd_f64 as fn(usize, u64) -> DMatrix<f64>,
        ),
        ("moderately-conditioned", moderately_conditioned_spd_f64),
        (
            "deliberately-ill-conditioned",
            deliberately_ill_conditioned_spd_f64,
        ),
    ] {
        for n in [16, 32, 48] {
            for seed in [0x82_1000, 0x82_2000, 0x82_3000] {
                let a = generate(n, seed);
                let packed_data = pack_lower_column_major(&a);
                let matrix = PackedSPD::from_vec(n, packed_data.clone()).unwrap();
                let exact = DMatrix::from_fn(n, 3, |row, col| {
                    (row as f64 * 0.13 - col as f64 * 0.19 + seed as f64 * 1e-7).cos()
                });
                let rhs = &a * &exact;
                let mut solved = rhs.as_slice().to_vec();
                matrix
                    .cholesky()
                    .unwrap()
                    .solve_many_in_place(&mut solved, 3)
                    .unwrap();
                let solved = DMatrix::from_column_slice(n, 3, &solved);
                let residual = matrix_rhs_residual(&a, &solved, &rhs);
                assert!(
                    residual < 2.0e-7,
                    "{} residual={residual:e}",
                    diagnostics(family, n, seed, &packed_data, &a, "extended nrhs=3")
                );
            }
        }
    }

    let singular = singular_psd_f64(12, 0x82_4000);
    assert!(singular.clone().symmetric_eigen().eigenvalues.min() <= 1.0e-12);

    let a = moderately_conditioned_spd_f64(32, 0x82_5000);
    let packed = PackedSymmetric::from_vec(32, pack_lower_column_major(&a)).unwrap();
    let basic = packed.eigendecomposition().unwrap();
    let dc = packed.eigendecomposition_divide_conquer().unwrap();
    for (&left, &right) in basic.eigenvalues.iter().zip(&dc.eigenvalues) {
        assert!((left - right).abs() <= 1.0e-8 * left.abs().max(right.abs()).max(1.0));
    }
    let vectors = DMatrix::from_column_slice(32, 32, basic.eigenvectors.as_ref().unwrap());
    assert_identity_close(
        &(vectors.transpose() * &vectors),
        Tolerance::for_scalar::<f64>(),
    );
    let diagonal = DMatrix::from_diagonal(&DVector::from_vec(basic.eigenvalues));
    assert_matrix_close(
        &a,
        &(&vectors * diagonal * vectors.transpose()),
        Tolerance {
            abs: 2.0e-7,
            rel: 2.0e-7,
        },
    );
}
