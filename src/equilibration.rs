use num_traits::Zero;

use crate::{
    PackedMatrixError, PackedSPD,
    backend::PositiveDefinitePackedEquilibration,
    factorization::checked_n,
    storage::{PackedStorage, PackedStorageMut},
};

/// Scaling information returned by LAPACK `xPPEQU`.
#[derive(Clone, Debug, PartialEq)]
pub struct Equilibration<R> {
    /// Factors `s[i]` such that the scaled entry is `s[i] * A(i,j) * s[j]`.
    pub scaling: Vec<R>,
    /// Ratio `min(s) / max(s)`; values near one indicate balanced scaling.
    pub condition_ratio: R,
    /// Largest diagonal magnitude reported by LAPACK.
    pub maximum_diagonal: R,
}

impl<T, S> PackedSPD<T, S>
where
    T: PositiveDefinitePackedEquilibration,
    T::Real: Zero,
    S: PackedStorage<T>,
{
    /// Computes packed positive-definite equilibration factors without modifying the matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension does not fit LAPACK, LAPACK rejects an
    /// argument, or a diagonal entry is non-positive.
    pub fn equilibration(&self) -> Result<Equilibration<T::Real>, PackedMatrixError> {
        let n = checked_n(self.dimension())?;
        let mut scaling = vec![T::Real::zero(); self.dimension()];
        let mut condition_ratio = [T::Real::zero()];
        let mut maximum_diagonal = T::Real::zero();
        let mut info = 0;
        unsafe {
            T::ppequ(
                b'L',
                n,
                self.as_slice(),
                &mut scaling,
                &mut condition_ratio,
                &mut maximum_diagonal,
                &mut info,
            )
        };
        if info < 0 {
            return Err(PackedMatrixError::LapackIllegalArgument { argument: -info });
        }
        if info > 0 {
            return Err(PackedMatrixError::NonPositiveDiagonal {
                index: info as usize,
            });
        }
        Ok(Equilibration {
            scaling,
            condition_ratio: condition_ratio[0],
            maximum_diagonal,
        })
    }
}

impl<T, S> PackedSPD<T, S>
where
    T: PositiveDefinitePackedEquilibration,
    T::Real: std::ops::Mul<Output = T::Real>,
    S: PackedStorageMut<T>,
{
    /// Applies caller-supplied equilibration factors directly to packed storage.
    ///
    /// Each logical entry becomes `s[i] * A[i,j] * s[j]`. Complex diagonal
    /// entries are canonicalized to real values.
    ///
    /// # Errors
    ///
    /// Returns [`PackedMatrixError::InvalidVectorLength`] unless `scaling`
    /// contains exactly one factor per row and column.
    pub fn apply_equilibration_in_place(
        &mut self,
        scaling: &[T::Real],
    ) -> Result<(), PackedMatrixError> {
        if scaling.len() != self.dimension() {
            return Err(PackedMatrixError::InvalidVectorLength {
                expected: self.dimension(),
                actual: scaling.len(),
            });
        }
        let n = self.dimension();
        let mut packed_index = 0;
        for col in 0..n {
            for row in col..n {
                let factor = scaling[row] * scaling[col];
                let value = self.as_mut_slice()[packed_index];
                let scaled = value.scale_by_real(factor);
                self.as_mut_slice()[packed_index] = if row == col {
                    scaled.canonicalize_diagonal()
                } else {
                    scaled
                };
                packed_index += 1;
            }
        }
        Ok(())
    }

    /// Computes and applies `xPPEQU` factors in place, returning the factors used.
    ///
    /// # Errors
    ///
    /// Returns the errors from [`Self::equilibration`] or
    /// [`Self::apply_equilibration_in_place`]. The matrix is modified only after
    /// factor computation succeeds.
    pub fn equilibrate_in_place(&mut self) -> Result<Equilibration<T::Real>, PackedMatrixError>
    where
        T::Real: Zero,
    {
        let factors = self.equilibration()?;
        self.apply_equilibration_in_place(&factors.scaling)?;
        Ok(factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn real_balanced_unbalanced_and_view() {
        let balanced = PackedSPD::from_vec(2, vec![4.0f64, 1.0, 4.0]).unwrap();
        let factors = balanced.equilibration().unwrap();
        assert_eq!(factors.scaling, vec![0.5, 0.5]);
        assert_eq!(factors.condition_ratio, 1.0);
        assert_eq!(factors.maximum_diagonal, 4.0);

        let storage = [1.0e-8f32, 0.0, 1.0e8];
        let view = PackedSPD::from_slice(2, &storage).unwrap();
        let factors = view.equilibration().unwrap();
        assert!(factors.condition_ratio < 1.0e-7);
        assert!(factors.scaling.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn nonpositive_diagonal_and_edge_sizes() {
        let bad = PackedSPD::from_vec(2, vec![1.0f64, 0.0, 0.0]).unwrap();
        assert_eq!(
            bad.equilibration(),
            Err(PackedMatrixError::NonPositiveDiagonal { index: 2 })
        );
        let one = PackedSPD::from_vec(1, vec![9.0f64])
            .unwrap()
            .equilibration()
            .unwrap();
        assert_eq!(one.scaling, vec![1.0 / 3.0]);
        let empty = PackedSPD::from_vec(0, Vec::<f64>::new())
            .unwrap()
            .equilibration()
            .unwrap();
        assert!(empty.scaling.is_empty());
    }

    #[test]
    fn real_and_complex_application_to_mutable_storage() {
        let mut real_storage = [4.0f64, 2.0, 9.0];
        let mut real = PackedSPD::from_slice_mut(2, &mut real_storage).unwrap();
        let factors = real.equilibrate_in_place().unwrap();
        assert!((real.get(0, 0).unwrap() - 1.0).abs() < 1e-12);
        assert!((real.get(1, 1).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(
            real.apply_equilibration_in_place(&factors.scaling[..1]),
            Err(PackedMatrixError::InvalidVectorLength {
                expected: 2,
                actual: 1
            })
        );

        let c = Complex64::new;
        let mut complex =
            PackedSPD::from_vec(2, vec![c(4.0, 7.0), c(1.0, -2.0), c(9.0, -5.0)]).unwrap();
        let factors = complex.equilibrate_in_place().unwrap();
        assert_eq!(factors.scaling, vec![0.5, 1.0 / 3.0]);
        assert!((complex.get(0, 0).unwrap().re - 1.0).abs() < 1e-12);
        assert!((complex.get(1, 1).unwrap().re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn complex32_equilibration() {
        let c = Complex32::new;
        let matrix = PackedSPD::from_vec(2, vec![c(4.0, 3.0), c(0.0, 1.0), c(16.0, -2.0)]).unwrap();
        let factors = matrix.equilibration().unwrap();
        assert_eq!(factors.scaling, vec![0.5, 0.25]);
        assert_eq!(factors.condition_ratio, 0.5);
        assert_eq!(factors.maximum_diagonal, 16.0);
    }
}
