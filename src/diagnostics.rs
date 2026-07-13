//! Diagnostics derived from packed triangular and factorized storage.

use crate::{
    Diagonal, PackedCholesky, PackedHermitianFactor, PackedLower, PackedSymmetricFactor,
    PackedUpper, storage::PackedStorage,
};
use num_complex::{Complex32, Complex64};

/// Counts of positive, negative, and zero eigenvalues of a symmetric or
/// Hermitian matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

/// A real determinant represented without forming its potentially overflowing
/// or underflowing magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignedLogDet<R> {
    /// `-1`, `0`, or `1`.
    pub sign: R,
    /// The natural logarithm of the absolute determinant, or negative infinity
    /// when `sign` is zero.
    pub log_abs: R,
}

fn lower_index(n: usize, row: usize, col: usize) -> usize {
    col * (2 * n - col + 1) / 2 + row - col
}

fn upper_index(row: usize, col: usize) -> usize {
    col * (col + 1) / 2 + row
}

macro_rules! impl_cholesky_diagnostics {
    ($scalar:ty, $real:ty, $diag:expr) => {
        impl<S: PackedStorage<$scalar>> PackedCholesky<$scalar, S> {
            /// Returns `ln(det(A))`, accumulated from the Cholesky diagonal.
            ///
            /// This logarithmic form is preferable when the determinant's
            /// magnitude may exceed or fall below the scalar's finite range.
            pub fn log_determinant(&self) -> $real {
                let data = self.data.as_slice();
                let mut sum: $real = 0.0;
                for j in 0..self.n {
                    let index = if self.uplo == b'L' {
                        lower_index(self.n, j, j)
                    } else {
                        upper_index(j, j)
                    };
                    let diagonal: $real = $diag(data[index]);
                    sum += diagonal.ln();
                }
                2.0 * sum
            }

            /// Returns `det(A)` from the Cholesky factor.
            ///
            /// This direct value may overflow to infinity or underflow to
            /// zero; use [`Self::log_determinant`] when magnitude is large.
            pub fn determinant(&self) -> $real {
                self.log_determinant().exp()
            }
        }
    };
}

impl_cholesky_diagnostics!(f32, f32, |x: f32| x);
impl_cholesky_diagnostics!(f64, f64, |x: f64| x);
impl_cholesky_diagnostics!(Complex32, f32, |x: Complex32| x.re);
impl_cholesky_diagnostics!(Complex64, f64, |x: Complex64| x.re);

macro_rules! impl_triangular_diagnostics {
    ($scalar:ty, $real:ty, $abs:expr) => {
        impl<S: PackedStorage<$scalar>> PackedLower<$scalar, S> {
            /// Returns the product of stored diagonal entries, or one for a
            /// unit-diagonal matrix.
            pub fn determinant(&self, diagonal: Diagonal) -> $scalar {
                if diagonal == Diagonal::Unit {
                    return <$scalar as num_traits::One>::one();
                }
                let mut determinant = <$scalar as num_traits::One>::one();
                for j in 0..self.dimension() {
                    determinant *= self.as_slice()[lower_index(self.dimension(), j, j)];
                }
                determinant
            }

            /// Returns the natural logarithm of the determinant magnitude.
            /// A singular non-unit triangular matrix returns negative infinity.
            pub fn log_abs_determinant(&self, diagonal: Diagonal) -> $real {
                if diagonal == Diagonal::Unit {
                    return 0.0;
                }
                let mut sum: $real = 0.0;
                for j in 0..self.dimension() {
                    let value: $real = $abs(self.as_slice()[lower_index(self.dimension(), j, j)]);
                    sum += value.ln();
                }
                sum
            }

            /// Tests exact singularity of the effective triangular diagonal.
            pub fn is_singular(&self, diagonal: Diagonal) -> bool {
                diagonal == Diagonal::NonUnit
                    && (0..self.dimension())
                        .any(|j| $abs(self.as_slice()[lower_index(self.dimension(), j, j)]) == 0.0)
            }
        }

        impl<S: PackedStorage<$scalar>> PackedUpper<$scalar, S> {
            /// Returns the product of stored diagonal entries, or one for a
            /// unit-diagonal matrix.
            pub fn determinant(&self, diagonal: Diagonal) -> $scalar {
                if diagonal == Diagonal::Unit {
                    return <$scalar as num_traits::One>::one();
                }
                let mut determinant = <$scalar as num_traits::One>::one();
                for j in 0..self.dimension() {
                    determinant *= self.as_slice()[upper_index(j, j)];
                }
                determinant
            }

            /// Returns the natural logarithm of the determinant magnitude.
            /// A singular non-unit triangular matrix returns negative infinity.
            pub fn log_abs_determinant(&self, diagonal: Diagonal) -> $real {
                if diagonal == Diagonal::Unit {
                    return 0.0;
                }
                let mut sum: $real = 0.0;
                for j in 0..self.dimension() {
                    let value: $real = $abs(self.as_slice()[upper_index(j, j)]);
                    sum += value.ln();
                }
                sum
            }

            /// Tests exact singularity of the effective triangular diagonal.
            pub fn is_singular(&self, diagonal: Diagonal) -> bool {
                diagonal == Diagonal::NonUnit
                    && (0..self.dimension())
                        .any(|j| $abs(self.as_slice()[upper_index(j, j)]) == 0.0)
            }
        }
    };
}

impl_triangular_diagnostics!(f32, f32, |x: f32| x.abs());
impl_triangular_diagnostics!(f64, f64, |x: f64| x.abs());
impl_triangular_diagnostics!(Complex32, f32, |x: Complex32| x.norm());
impl_triangular_diagnostics!(Complex64, f64, |x: Complex64| x.norm());

fn add_sign<R: PartialOrd + From<i8> + std::ops::Mul<Output = R>>(
    inertia: &mut Inertia,
    value: R,
    tolerance: R,
) {
    if value > tolerance {
        inertia.positive += 1;
    } else if value < R::from(-1) * tolerance {
        inertia.negative += 1;
    } else {
        inertia.zero += 1;
    }
}

macro_rules! impl_symmetric_diagnostics {
    ($real:ty) => {
        impl<S: PackedStorage<$real>> PackedSymmetricFactor<$real, S> {
            fn d_blocks(&self) -> Vec<($real, Option<($real, $real)>)> {
                let data = self.data.as_slice();
                let mut blocks = Vec::new();
                if self.uplo == b'L' {
                    let mut k = 0;
                    while k < self.n {
                        let a = data[lower_index(self.n, k, k)];
                        if self.pivots[k] < 0 {
                            let b = data[lower_index(self.n, k + 1, k)];
                            let c = data[lower_index(self.n, k + 1, k + 1)];
                            blocks.push((a, Some((b, c))));
                            k += 2;
                        } else {
                            blocks.push((a, None));
                            k += 1;
                        }
                    }
                } else {
                    let mut end = self.n;
                    while end > 0 {
                        let k = end - 1;
                        if self.pivots[k] < 0 {
                            let a = data[upper_index(k - 1, k - 1)];
                            let b = data[upper_index(k - 1, k)];
                            let c = data[upper_index(k, k)];
                            blocks.push((a, Some((b, c))));
                            end -= 2;
                        } else {
                            blocks.push((data[upper_index(k, k)], None));
                            end -= 1;
                        }
                    }
                }
                blocks
            }

            /// Returns determinant sign and log-magnitude from the
            /// Bunch-Kaufman diagonal blocks.
            pub fn slogdet(&self) -> SignedLogDet<$real> {
                let mut sign: $real = 1.0;
                let mut log_abs: $real = 0.0;
                for (a, pair) in self.d_blocks() {
                    if let Some((b, c)) = pair {
                        let scale = a.abs().max(b.abs()).max(c.abs());
                        if scale == 0.0 {
                            return SignedLogDet {
                                sign: 0.0,
                                log_abs: <$real>::NEG_INFINITY,
                            };
                        }
                        let scaled = (a / scale) * (c / scale) - (b / scale) * (b / scale);
                        if scaled == 0.0 {
                            return SignedLogDet {
                                sign: 0.0,
                                log_abs: <$real>::NEG_INFINITY,
                            };
                        }
                        sign *= scaled.signum();
                        log_abs += scaled.abs().ln() + 2.0 * scale.ln();
                    } else if a == 0.0 {
                        return SignedLogDet {
                            sign: 0.0,
                            log_abs: <$real>::NEG_INFINITY,
                        };
                    } else {
                        sign *= a.signum();
                        log_abs += a.abs().ln();
                    }
                }
                SignedLogDet { sign, log_abs }
            }

            /// Returns inertia using exact-zero classification of each
            /// Bunch-Kaufman diagonal block.
            pub fn inertia(&self) -> Inertia {
                let mut result = Inertia::default();
                for (a, pair) in self.d_blocks() {
                    if let Some((b, c)) = pair {
                        let scale = a.abs().max(b.abs()).max(c.abs());
                        if scale == 0.0 {
                            result.zero += 2;
                            continue;
                        }
                        let determinant = (a / scale) * (c / scale) - (b / scale) * (b / scale);
                        if determinant < 0.0 {
                            result.positive += 1;
                            result.negative += 1;
                        } else if determinant > 0.0 {
                            add_sign(&mut result, a + c, 0.0);
                            add_sign(&mut result, a + c, 0.0);
                        } else {
                            result.zero += 1;
                            add_sign(&mut result, a + c, 0.0);
                        }
                    } else {
                        add_sign(&mut result, a, 0.0);
                    }
                }
                result
            }

            /// Returns inertia with eigenvalues whose absolute value is at
            /// most `abs(tolerance)` classified as zero.
            pub fn inertia_with_tolerance(&self, tolerance: $real) -> Inertia {
                let tolerance = tolerance.abs();
                let mut result = Inertia::default();
                for (a, pair) in self.d_blocks() {
                    if let Some((b, c)) = pair {
                        let half_trace = (a + c) / 2.0;
                        let radius = ((a - c) / 2.0).hypot(b);
                        add_sign(&mut result, half_trace + radius, tolerance);
                        add_sign(&mut result, half_trace - radius, tolerance);
                    } else {
                        add_sign(&mut result, a, tolerance);
                    }
                }
                result
            }

            /// Tests exact singularity of the block diagonal factor `D`.
            pub fn is_singular(&self) -> bool {
                self.inertia().zero != 0
            }
        }
    };
}

impl_symmetric_diagnostics!(f32);
impl_symmetric_diagnostics!(f64);

macro_rules! impl_hermitian_diagnostics {
    ($complex:ty, $real:ty) => {
        impl<S: PackedStorage<$complex>> PackedHermitianFactor<$complex, S> {
            fn d_blocks(&self) -> Vec<($real, Option<($real, $real)>)> {
                let data = self.data.as_slice();
                let mut blocks = Vec::new();
                if self.uplo == b'L' {
                    let mut k = 0;
                    while k < self.n {
                        let a = data[lower_index(self.n, k, k)].re;
                        if self.pivots[k] < 0 {
                            let b = data[lower_index(self.n, k + 1, k)].norm();
                            let c = data[lower_index(self.n, k + 1, k + 1)].re;
                            blocks.push((a, Some((b, c))));
                            k += 2;
                        } else {
                            blocks.push((a, None));
                            k += 1;
                        }
                    }
                } else {
                    let mut end = self.n;
                    while end > 0 {
                        let k = end - 1;
                        if self.pivots[k] < 0 {
                            let a = data[upper_index(k - 1, k - 1)].re;
                            let b = data[upper_index(k - 1, k)].norm();
                            let c = data[upper_index(k, k)].re;
                            blocks.push((a, Some((b, c))));
                            end -= 2;
                        } else {
                            blocks.push((data[upper_index(k, k)].re, None));
                            end -= 1;
                        }
                    }
                }
                blocks
            }

            /// Returns the real determinant sign and log-magnitude from the
            /// Hermitian Bunch-Kaufman diagonal blocks.
            pub fn slogdet(&self) -> SignedLogDet<$real> {
                let mut sign: $real = 1.0;
                let mut log_abs: $real = 0.0;
                for (a, pair) in self.d_blocks() {
                    if let Some((b, c)) = pair {
                        let scale = a.abs().max(b.abs()).max(c.abs());
                        if scale == 0.0 {
                            return SignedLogDet {
                                sign: 0.0,
                                log_abs: <$real>::NEG_INFINITY,
                            };
                        }
                        let scaled = (a / scale) * (c / scale) - (b / scale) * (b / scale);
                        if scaled == 0.0 {
                            return SignedLogDet {
                                sign: 0.0,
                                log_abs: <$real>::NEG_INFINITY,
                            };
                        }
                        sign *= scaled.signum();
                        log_abs += scaled.abs().ln() + 2.0 * scale.ln();
                    } else if a == 0.0 {
                        return SignedLogDet {
                            sign: 0.0,
                            log_abs: <$real>::NEG_INFINITY,
                        };
                    } else {
                        sign *= a.signum();
                        log_abs += a.abs().ln();
                    }
                }
                SignedLogDet { sign, log_abs }
            }

            /// Returns inertia using exact-zero classification.
            pub fn inertia(&self) -> Inertia {
                let mut result = Inertia::default();
                for (a, pair) in self.d_blocks() {
                    if let Some((b, c)) = pair {
                        let scale = a.abs().max(b.abs()).max(c.abs());
                        if scale == 0.0 {
                            result.zero += 2;
                            continue;
                        }
                        let determinant = (a / scale) * (c / scale) - (b / scale) * (b / scale);
                        if determinant < 0.0 {
                            result.positive += 1;
                            result.negative += 1;
                        } else if determinant > 0.0 {
                            add_sign(&mut result, a + c, 0.0);
                            add_sign(&mut result, a + c, 0.0);
                        } else {
                            result.zero += 1;
                            add_sign(&mut result, a + c, 0.0);
                        }
                    } else {
                        add_sign(&mut result, a, 0.0);
                    }
                }
                result
            }

            /// Returns inertia with eigenvalues whose absolute value is at
            /// most `abs(tolerance)` classified as zero.
            pub fn inertia_with_tolerance(&self, tolerance: $real) -> Inertia {
                let tolerance = tolerance.abs();
                let mut result = Inertia::default();
                for (a, pair) in self.d_blocks() {
                    if let Some((b, c)) = pair {
                        let half_trace = (a + c) / 2.0;
                        let radius = ((a - c) / 2.0).hypot(b);
                        add_sign(&mut result, half_trace + radius, tolerance);
                        add_sign(&mut result, half_trace - radius, tolerance);
                    } else {
                        add_sign(&mut result, a, tolerance);
                    }
                }
                result
            }

            /// Tests exact singularity of the block diagonal factor `D`.
            pub fn is_singular(&self) -> bool {
                self.inertia().zero != 0
            }
        }
    };
}

impl_hermitian_diagnostics!(Complex32, f32);
impl_hermitian_diagnostics!(Complex64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PackedHermitian, PackedSPD, PackedSymmetric};

    #[test]
    fn cholesky_logdet_is_stable_and_real_for_hpd() {
        let real = PackedSPD::from_vec(2, vec![4.0f64, 0.0, 9.0])
            .unwrap()
            .cholesky()
            .unwrap();
        assert!((real.log_determinant() - 36.0f64.ln()).abs() < 1e-12);
        assert!((real.determinant() - 36.0).abs() < 1e-12);

        let huge = PackedSPD::from_vec(2, vec![1e200f64, 0.0, 1e200])
            .unwrap()
            .cholesky()
            .unwrap();
        assert!(huge.log_determinant().is_finite());
        assert!(huge.determinant().is_infinite());

        let c = Complex64::new;
        let complex = PackedSPD::from_vec(2, vec![c(4.0, 0.0), c(0.0, 0.0), c(9.0, 0.0)])
            .unwrap()
            .cholesky()
            .unwrap();
        assert!((complex.log_determinant() - 36.0f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn triangular_real_complex_unit_and_singular() {
        let lower = PackedLower::from_vec(2, vec![2.0f64, 7.0, -3.0]).unwrap();
        assert_eq!(lower.determinant(Diagonal::NonUnit), -6.0);
        assert!((lower.log_abs_determinant(Diagonal::NonUnit) - 6.0f64.ln()).abs() < 1e-12);
        assert_eq!(lower.determinant(Diagonal::Unit), 1.0);

        let c = Complex64::new;
        let upper = PackedUpper::from_vec(2, vec![c(1.0, 1.0), c(4.0, 0.0), c(0.0, 2.0)]).unwrap();
        assert_eq!(upper.determinant(Diagonal::NonUnit), c(-2.0, 2.0));
        assert!(!upper.is_singular(Diagonal::NonUnit));
        let singular = PackedUpper::from_vec(1, vec![c(0.0, 0.0)]).unwrap();
        assert!(singular.is_singular(Diagonal::NonUnit));
        assert_eq!(
            singular.log_abs_determinant(Diagonal::NonUnit),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn symmetric_inertia_and_two_by_two_pivot() {
        let positive = PackedSymmetric::from_vec(2, vec![2.0f64, 0.0, 3.0])
            .unwrap()
            .factorize()
            .unwrap();
        assert_eq!(
            positive.inertia(),
            Inertia {
                positive: 2,
                negative: 0,
                zero: 0
            }
        );

        let negative = PackedSymmetric::from_vec(2, vec![-2.0f64, 0.0, -3.0])
            .unwrap()
            .factorize()
            .unwrap();
        assert_eq!(
            negative.inertia(),
            Inertia {
                positive: 0,
                negative: 2,
                zero: 0
            }
        );
        assert_eq!(negative.slogdet().sign, 1.0);

        let indefinite = PackedSymmetric::from_vec(2, vec![0.0f64, 1.0, 0.0])
            .unwrap()
            .factorize()
            .unwrap();
        assert!(indefinite.pivots()[0] < 0);
        assert_eq!(
            indefinite.inertia(),
            Inertia {
                positive: 1,
                negative: 1,
                zero: 0
            }
        );
        let slog = indefinite.slogdet();
        assert_eq!(slog.sign, -1.0);
        assert_eq!(slog.log_abs, 0.0);
    }

    #[test]
    fn hermitian_inertia_and_explicit_zero_classification() {
        let c = Complex64::new;
        let factor = PackedHermitian::from_vec(2, vec![c(0.0, 0.0), c(0.0, 1.0), c(0.0, 0.0)])
            .unwrap()
            .factorize()
            .unwrap();
        assert!(factor.pivots()[0] < 0);
        assert_eq!(
            factor.inertia(),
            Inertia {
                positive: 1,
                negative: 1,
                zero: 0
            }
        );
        assert_eq!(
            factor.slogdet(),
            SignedLogDet {
                sign: -1.0,
                log_abs: 0.0
            }
        );

        let synthetic = PackedSymmetricFactor {
            n: 2,
            data: vec![1.0f64, 0.0, 0.0],
            pivots: vec![1, 2],
            uplo: b'L',
            marker: std::marker::PhantomData,
        };
        assert_eq!(
            synthetic.inertia(),
            Inertia {
                positive: 1,
                negative: 0,
                zero: 1
            }
        );
        assert!(synthetic.is_singular());
        assert_eq!(synthetic.slogdet().sign, 0.0);
        assert_eq!(
            synthetic.inertia_with_tolerance(2.0),
            Inertia {
                positive: 0,
                negative: 0,
                zero: 2
            }
        );
    }

    #[test]
    fn upper_factor_layouts_decode_two_by_two_blocks() {
        let symmetric = PackedSymmetricFactor {
            n: 2,
            data: vec![0.0f64, 1.0, 0.0],
            pivots: vec![-1, -1],
            uplo: b'U',
            marker: std::marker::PhantomData,
        };
        assert_eq!(
            symmetric.inertia(),
            Inertia {
                positive: 1,
                negative: 1,
                zero: 0
            }
        );
        assert_eq!(symmetric.slogdet().sign, -1.0);

        let c = Complex64::new;
        let hermitian = PackedHermitianFactor {
            n: 2,
            data: vec![c(0.0, 0.0), c(0.0, 1.0), c(0.0, 0.0)],
            pivots: vec![-1, -1],
            uplo: b'U',
            marker: std::marker::PhantomData,
        };
        assert_eq!(
            hermitian.inertia(),
            Inertia {
                positive: 1,
                negative: 1,
                zero: 0
            }
        );
        assert_eq!(hermitian.slogdet().sign, -1.0);
    }
}
