use crate::{
    backend::{HermitianPackedRankUpdate, RealSymmetricPackedRankUpdate},
    factorization::checked_n,
    storage::PackedStorageMut,
    PackedHermitian, PackedMatrixError, PackedSymmetric,
};

fn validate_strided_vector<T>(
    n: usize,
    vector: &[T],
    increment: i32,
) -> Result<(), PackedMatrixError> {
    if increment == 0 {
        return Err(PackedMatrixError::InvalidIncrement { increment });
    }
    let stride = increment.unsigned_abs() as usize;
    let expected = if n == 0 {
        0
    } else {
        n.checked_sub(1)
            .and_then(|count| count.checked_mul(stride))
            .and_then(|span| span.checked_add(1))
            .ok_or(PackedMatrixError::DimensionOverflow { n })?
    };
    if vector.len() < expected {
        return Err(PackedMatrixError::InvalidVectorLength {
            expected,
            actual: vector.len(),
        });
    }
    Ok(())
}

impl<T, S> PackedSymmetric<T, S>
where
    T: RealSymmetricPackedRankUpdate,
    S: PackedStorageMut<T>,
{
    /// Performs `A := A + alpha*x*x^T` in packed storage.
    pub fn rank1_update_in_place(&mut self, alpha: T, x: &[T]) -> Result<(), PackedMatrixError> {
        self.rank1_update_strided_in_place(alpha, x, 1)
    }

    /// Performs a symmetric rank-1 update using a nonzero BLAS increment.
    pub fn rank1_update_strided_in_place(
        &mut self,
        alpha: T,
        x: &[T],
        incx: i32,
    ) -> Result<(), PackedMatrixError> {
        validate_strided_vector(self.dimension(), x, incx)?;
        let n = checked_n(self.dimension())?;
        unsafe { T::spr(b'L', n, alpha, x, incx, self.as_mut_slice()) };
        Ok(())
    }

    /// Performs `A := A + alpha*x*y^T + alpha*y*x^T` in packed storage.
    pub fn rank2_update_in_place(
        &mut self,
        alpha: T,
        x: &[T],
        y: &[T],
    ) -> Result<(), PackedMatrixError> {
        self.rank2_update_strided_in_place(alpha, x, 1, y, 1)
    }

    /// Performs a symmetric rank-2 update using nonzero BLAS increments.
    pub fn rank2_update_strided_in_place(
        &mut self,
        alpha: T,
        x: &[T],
        incx: i32,
        y: &[T],
        incy: i32,
    ) -> Result<(), PackedMatrixError> {
        validate_strided_vector(self.dimension(), x, incx)?;
        validate_strided_vector(self.dimension(), y, incy)?;
        let n = checked_n(self.dimension())?;
        unsafe { T::spr2(b'L', n, alpha, x, incx, y, incy, self.as_mut_slice()) };
        Ok(())
    }
}

impl<T, S> PackedHermitian<T, S>
where
    T: HermitianPackedRankUpdate,
    S: PackedStorageMut<T>,
{
    /// Performs `A := A + alpha*x*x^H`; `alpha` is real by construction.
    pub fn rank1_update_in_place(
        &mut self,
        alpha: T::Real,
        x: &[T],
    ) -> Result<(), PackedMatrixError> {
        self.rank1_update_strided_in_place(alpha, x, 1)
    }

    /// Performs a Hermitian rank-1 update using a nonzero BLAS increment.
    pub fn rank1_update_strided_in_place(
        &mut self,
        alpha: T::Real,
        x: &[T],
        incx: i32,
    ) -> Result<(), PackedMatrixError> {
        validate_strided_vector(self.dimension(), x, incx)?;
        let n = checked_n(self.dimension())?;
        unsafe { T::hpr(b'L', n, alpha, x, incx, self.as_mut_slice()) };
        Ok(())
    }

    /// Performs `A := A + alpha*x*y^H + conj(alpha)*y*x^H`.
    pub fn rank2_update_in_place(
        &mut self,
        alpha: T,
        x: &[T],
        y: &[T],
    ) -> Result<(), PackedMatrixError> {
        self.rank2_update_strided_in_place(alpha, x, 1, y, 1)
    }

    /// Performs a Hermitian rank-2 update using nonzero BLAS increments.
    pub fn rank2_update_strided_in_place(
        &mut self,
        alpha: T,
        x: &[T],
        incx: i32,
        y: &[T],
        incy: i32,
    ) -> Result<(), PackedMatrixError> {
        validate_strided_vector(self.dimension(), x, incx)?;
        validate_strided_vector(self.dimension(), y, incy)?;
        let n = checked_n(self.dimension())?;
        unsafe { T::hpr2(b'L', n, alpha, x, incx, y, incy, self.as_mut_slice()) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }

    #[test]
    fn real_owned_rank1_rank2_and_zero() {
        let mut a = PackedSymmetric::from_vec(2, vec![1.0f64, 2.0, 3.0]).unwrap();
        a.rank1_update_in_place(2.0, &[1.0, -1.0]).unwrap();
        close(a.get(0, 0).unwrap(), 3.0);
        close(a.get(0, 1).unwrap(), 0.0);
        close(a.get(1, 1).unwrap(), 5.0);
        a.rank2_update_in_place(0.5, &[1.0, 2.0], &[3.0, 4.0])
            .unwrap();
        close(a.get(0, 0).unwrap(), 6.0);
        close(a.get(0, 1).unwrap(), 5.0);
        close(a.get(1, 1).unwrap(), 13.0);
        let before = a.as_slice().to_vec();
        a.rank1_update_in_place(0.0, &[9.0, 8.0]).unwrap();
        assert_eq!(a.as_slice(), before);
    }

    #[test]
    fn real_mutable_view_stride_and_validation() {
        let mut storage = [0.0f32; 3];
        let mut a = PackedSymmetric::from_slice_mut(2, &mut storage).unwrap();
        a.rank1_update_strided_in_place(1.0, &[2.0, 99.0, 3.0], 2)
            .unwrap();
        assert_eq!(a.get(0, 0).unwrap(), 4.0);
        assert_eq!(a.get(0, 1).unwrap(), 6.0);
        assert_eq!(a.get(1, 1).unwrap(), 9.0);
        assert!(matches!(
            a.rank1_update_strided_in_place(1.0, &[1.0], 0),
            Err(PackedMatrixError::InvalidIncrement { .. })
        ));
        assert!(matches!(
            a.rank2_update_strided_in_place(1.0, &[1.0], 2, &[1.0, 2.0], 1),
            Err(PackedMatrixError::InvalidVectorLength { .. })
        ));
    }

    #[test]
    fn negative_stride_one_and_empty() {
        let mut a = PackedSymmetric::from_vec(2, vec![0.0f64; 3]).unwrap();
        a.rank1_update_strided_in_place(1.0, &[2.0, 3.0], -1)
            .unwrap();
        close(a.get(0, 0).unwrap(), 9.0);
        close(a.get(0, 1).unwrap(), 6.0);
        close(a.get(1, 1).unwrap(), 4.0);
        let mut one = PackedSymmetric::from_vec(1, vec![2.0f64]).unwrap();
        one.rank1_update_in_place(3.0, &[2.0]).unwrap();
        close(one.get(0, 0).unwrap(), 14.0);
        let mut empty = PackedSymmetric::from_vec(0, Vec::<f64>::new()).unwrap();
        empty.rank1_update_in_place(1.0, &[]).unwrap();
    }

    #[test]
    fn complex_hermitian_rank_updates_and_real_diagonal() {
        let c = Complex64::new;
        let mut a =
            PackedHermitian::from_vec(2, vec![c(1.0, 0.0), c(0.0, 0.0), c(2.0, 0.0)]).unwrap();
        let x = [c(1.0, 1.0), c(2.0, -1.0)];
        a.rank1_update_in_place(0.5, &x).unwrap();
        close(a.get(0, 0).unwrap().im, 0.0);
        close(a.get(1, 1).unwrap().im, 0.0);
        let y = [c(-1.0, 2.0), c(0.5, 1.0)];
        a.rank2_update_in_place(c(0.25, -0.5), &x, &y).unwrap();
        close(a.get(0, 0).unwrap().im, 0.0);
        close(a.get(1, 1).unwrap().im, 0.0);
        let upper = a.get(0, 1).unwrap();
        let lower = a.get(1, 0).unwrap();
        assert!((upper - lower.conj()).norm() < 1e-12);
    }

    #[test]
    fn complex32_mutable_view_strided() {
        let c = Complex32::new;
        let mut storage = [c(0.0, 0.0); 3];
        let mut a = PackedHermitian::from_slice_mut(2, &mut storage).unwrap();
        a.rank2_update_strided_in_place(
            c(1.0, 0.5),
            &[c(1.0, 0.0), c(9.0, 9.0), c(2.0, 1.0)],
            2,
            &[c(0.0, 1.0), c(8.0, 8.0), c(1.0, -1.0)],
            2,
        )
        .unwrap();
        assert_eq!(a.get(0, 0).unwrap().im, 0.0);
        assert_eq!(a.get(1, 1).unwrap().im, 0.0);
    }

    #[test]
    fn spd_hpd_are_downgraded_before_unrestricted_updates() {
        let spd = crate::PackedSPD::from_vec(1, vec![2.0f64]).unwrap();
        let mut symmetric = spd.into_symmetric();
        symmetric.rank1_update_in_place(-3.0, &[1.0]).unwrap();
        assert_eq!(symmetric.get(0, 0).unwrap(), -1.0);

        let hpd = crate::PackedSPD::from_vec(1, vec![Complex64::new(2.0, 0.0)]).unwrap();
        let mut hermitian = hpd.into_hermitian();
        hermitian
            .rank1_update_in_place(-3.0, &[Complex64::new(1.0, 0.0)])
            .unwrap();
        assert_eq!(hermitian.get(0, 0).unwrap(), Complex64::new(-1.0, 0.0));
    }
}
