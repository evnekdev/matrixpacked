use crate::{
    PackedHermitian, PackedMatrixError, PackedSymmetric,
    backend::{HermitianPackedTridiagonalBackend, SymmetricPackedTridiagonalBackend},
    factorization::{check_info, checked_n},
    storage::{PackedStorage, PackedStorageMut},
};
use num_traits::Zero;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplySide {
    Left,
    Right,
}
impl ApplySide {
    fn lapack(self) -> u8 {
        match self {
            Self::Left => b'L',
            Self::Right => b'R',
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrthogonalOperation {
    None,
    Transpose,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitaryOperation {
    None,
    ConjugateTranspose,
}

#[derive(Clone, Debug)]
pub struct SymmetricPackedTridiagonal<T, S = Vec<T>> {
    n: usize,
    data: S,
    diagonal: Vec<T>,
    off_diagonal: Vec<T>,
    tau: Vec<T>,
}
#[derive(Clone, Debug)]
pub struct HermitianPackedTridiagonal<T: crate::LapackScalar, S = Vec<T>> {
    n: usize,
    data: S,
    diagonal: Vec<T::Real>,
    off_diagonal: Vec<T::Real>,
    tau: Vec<T>,
}

fn dense_len(rows: usize, cols: usize) -> Result<usize, PackedMatrixError> {
    rows.checked_mul(cols)
        .ok_or(PackedMatrixError::DimensionOverflow { n: rows.max(cols) })
}
fn validate_dense<T>(
    order: usize,
    side: ApplySide,
    rows: usize,
    cols: usize,
    ldc: usize,
    c: &[T],
) -> Result<i32, PackedMatrixError> {
    if (side == ApplySide::Left && rows != order) || (side == ApplySide::Right && cols != order) {
        return Err(PackedMatrixError::DimensionMismatch {
            left: if side == ApplySide::Left { rows } else { cols },
            right: order,
        });
    }
    if ldc < rows.max(1) {
        return Err(PackedMatrixError::InvalidLeadingDimension {
            minimum: rows.max(1),
            actual: ldc,
        });
    }
    let expected = dense_len(ldc, cols)?;
    if c.len() != expected {
        return Err(PackedMatrixError::InvalidVectorLength {
            expected,
            actual: c.len(),
        });
    }
    checked_n(ldc)
}

impl<T, S> SymmetricPackedTridiagonal<T, S>
where
    T: SymmetricPackedTridiagonalBackend,
    S: PackedStorage<T>,
{
    pub fn dimension(&self) -> usize {
        self.n
    }
    pub fn packed_reflectors(&self) -> &[T] {
        self.data.as_slice()
    }
    pub fn diagonal(&self) -> &[T] {
        &self.diagonal
    }
    pub fn off_diagonal(&self) -> &[T] {
        &self.off_diagonal
    }
    pub fn tau(&self) -> &[T] {
        &self.tau
    }
    pub fn generate_q(&self) -> Result<Vec<T>, PackedMatrixError> {
        let mut q = vec![T::zero(); dense_len(self.n, self.n)?];
        let mut w = vec![T::zero(); self.n.saturating_sub(1)];
        let mut info = 0;
        unsafe {
            T::opgtr(
                b'L',
                checked_n(self.n)?,
                self.data.as_slice(),
                &self.tau,
                &mut q,
                checked_n(self.n.max(1))?,
                &mut w,
                &mut info,
            )
        };
        check_info(info, "orthogonal matrix generation failed")?;
        Ok(q)
    }
    pub fn apply_q_in_place(
        &self,
        side: ApplySide,
        operation: OrthogonalOperation,
        rows: usize,
        cols: usize,
        ldc: usize,
        c: &mut [T],
    ) -> Result<(), PackedMatrixError> {
        let ld = validate_dense(self.n, side, rows, cols, ldc, c)?;
        let mut w = vec![T::zero(); if side == ApplySide::Left { cols } else { rows }];
        let mut info = 0;
        unsafe {
            T::opmtr(
                side.lapack(),
                b'L',
                match operation {
                    OrthogonalOperation::None => b'N',
                    OrthogonalOperation::Transpose => b'T',
                },
                checked_n(rows)?,
                checked_n(cols)?,
                self.data.as_slice(),
                &self.tau,
                c,
                ld,
                &mut w,
                &mut info,
            )
        };
        check_info(info, "orthogonal transformation application failed")
    }
}
impl<T, S> HermitianPackedTridiagonal<T, S>
where
    T: HermitianPackedTridiagonalBackend,
    T::Real: Zero,
    S: PackedStorage<T>,
{
    pub fn dimension(&self) -> usize {
        self.n
    }
    pub fn packed_reflectors(&self) -> &[T] {
        self.data.as_slice()
    }
    pub fn diagonal(&self) -> &[T::Real] {
        &self.diagonal
    }
    pub fn off_diagonal(&self) -> &[T::Real] {
        &self.off_diagonal
    }
    pub fn tau(&self) -> &[T] {
        &self.tau
    }
    pub fn generate_q(&self) -> Result<Vec<T>, PackedMatrixError> {
        let mut q = vec![T::zero(); dense_len(self.n, self.n)?];
        let mut w = vec![T::zero(); self.n.saturating_sub(1)];
        let mut info = 0;
        unsafe {
            T::upgtr(
                b'L',
                checked_n(self.n)?,
                self.data.as_slice(),
                &self.tau,
                &mut q,
                checked_n(self.n.max(1))?,
                &mut w,
                &mut info,
            )
        };
        check_info(info, "unitary matrix generation failed")?;
        Ok(q)
    }
    pub fn apply_q_in_place(
        &self,
        side: ApplySide,
        operation: UnitaryOperation,
        rows: usize,
        cols: usize,
        ldc: usize,
        c: &mut [T],
    ) -> Result<(), PackedMatrixError> {
        let ld = validate_dense(self.n, side, rows, cols, ldc, c)?;
        let mut w = vec![T::zero(); if side == ApplySide::Left { cols } else { rows }];
        let mut info = 0;
        unsafe {
            T::upmtr(
                side.lapack(),
                b'L',
                match operation {
                    UnitaryOperation::None => b'N',
                    UnitaryOperation::ConjugateTranspose => b'C',
                },
                checked_n(rows)?,
                checked_n(cols)?,
                self.data.as_slice(),
                &self.tau,
                c,
                ld,
                &mut w,
                &mut info,
            )
        };
        check_info(info, "unitary transformation application failed")
    }
}

fn reduce_symmetric<T, S>(
    n: usize,
    mut data: S,
) -> Result<SymmetricPackedTridiagonal<T, S>, PackedMatrixError>
where
    T: SymmetricPackedTridiagonalBackend,
    S: PackedStorageMut<T>,
{
    let mut d = vec![T::zero(); n];
    let mut e = vec![T::zero(); n.saturating_sub(1)];
    let mut tau = vec![T::zero(); n.saturating_sub(1)];
    let mut info = 0;
    unsafe {
        T::sptrd(
            b'L',
            checked_n(n)?,
            data.as_mut_slice(),
            &mut d,
            &mut e,
            &mut tau,
            &mut info,
        )
    };
    check_info(info, "symmetric packed tridiagonal reduction failed")?;
    Ok(SymmetricPackedTridiagonal {
        n,
        data,
        diagonal: d,
        off_diagonal: e,
        tau,
    })
}
fn reduce_hermitian<T, S>(
    n: usize,
    mut data: S,
) -> Result<HermitianPackedTridiagonal<T, S>, PackedMatrixError>
where
    T: HermitianPackedTridiagonalBackend,
    T::Real: Zero,
    S: PackedStorageMut<T>,
{
    let mut d = vec![T::Real::zero(); n];
    let mut e = vec![T::Real::zero(); n.saturating_sub(1)];
    let mut tau = vec![T::zero(); n.saturating_sub(1)];
    let mut info = 0;
    unsafe {
        T::hptrd(
            b'L',
            checked_n(n)?,
            data.as_mut_slice(),
            &mut d,
            &mut e,
            &mut tau,
            &mut info,
        )
    };
    check_info(info, "Hermitian packed tridiagonal reduction failed")?;
    Ok(HermitianPackedTridiagonal {
        n,
        data,
        diagonal: d,
        off_diagonal: e,
        tau,
    })
}

impl<T, S> PackedSymmetric<T, S>
where
    T: SymmetricPackedTridiagonalBackend,
    S: PackedStorage<T>,
{
    pub fn tridiagonal_reduction(
        &self,
    ) -> Result<SymmetricPackedTridiagonal<T>, PackedMatrixError> {
        reduce_symmetric(self.dimension(), self.as_slice().to_vec())
    }
}
impl<T, S> PackedSymmetric<T, S>
where
    T: SymmetricPackedTridiagonalBackend,
    S: PackedStorageMut<T>,
{
    pub fn tridiagonal_reduction_in_place(
        self,
    ) -> Result<SymmetricPackedTridiagonal<T, S>, PackedMatrixError> {
        let n = self.dimension();
        reduce_symmetric(n, self.into_storage())
    }
}
impl<T> PackedSymmetric<T, Vec<T>>
where
    T: SymmetricPackedTridiagonalBackend,
{
    pub fn into_tridiagonal_reduction(
        self,
    ) -> Result<SymmetricPackedTridiagonal<T>, PackedMatrixError> {
        self.tridiagonal_reduction_in_place()
    }
}
impl<T, S> PackedHermitian<T, S>
where
    T: HermitianPackedTridiagonalBackend,
    T::Real: Zero,
    S: PackedStorage<T>,
{
    pub fn tridiagonal_reduction(
        &self,
    ) -> Result<HermitianPackedTridiagonal<T>, PackedMatrixError> {
        reduce_hermitian(self.dimension(), self.as_slice().to_vec())
    }
}
impl<T, S> PackedHermitian<T, S>
where
    T: HermitianPackedTridiagonalBackend,
    T::Real: Zero,
    S: PackedStorageMut<T>,
{
    pub fn tridiagonal_reduction_in_place(
        self,
    ) -> Result<HermitianPackedTridiagonal<T, S>, PackedMatrixError> {
        let n = self.dimension();
        reduce_hermitian(n, self.into_storage())
    }
}
impl<T> PackedHermitian<T, Vec<T>>
where
    T: HermitianPackedTridiagonalBackend,
    T::Real: Zero,
{
    pub fn into_tridiagonal_reduction(
        self,
    ) -> Result<HermitianPackedTridiagonal<T>, PackedMatrixError> {
        self.tridiagonal_reduction_in_place()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn real_reconstruction_generation_application_and_view() {
        let original = [4.0f64, 1.0, 2.0, 3.0, 0.5, 2.0];
        let a = PackedSymmetric::from_vec(3, original.to_vec()).unwrap();
        let r = a.tridiagonal_reduction().unwrap();
        let q = r.generate_q().unwrap();
        let mut t = [0.0; 9];
        for i in 0..3 {
            t[i + i * 3] = r.diagonal()[i];
            if i < 2 {
                t[i + (i + 1) * 3] = r.off_diagonal()[i];
                t[i + 1 + i * 3] = r.off_diagonal()[i];
            }
        }
        let mut reconstructed = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        reconstructed[row + col * 3] +=
                            q[row + k * 3] * t[k + l * 3] * q[col + l * 3];
                    }
                }
            }
        }
        let dense = [4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 2.0];
        for (a, b) in reconstructed.iter().zip(dense) {
            assert!((*a - b).abs() < 1e-10);
        }
        let mut left = vec![0.0; 9];
        for i in 0..3 {
            left[i + i * 3] = 1.0;
        }
        r.apply_q_in_place(
            ApplySide::Left,
            OrthogonalOperation::None,
            3,
            3,
            3,
            &mut left,
        )
        .unwrap();
        for (a, b) in left.iter().zip(&q) {
            assert!((*a - *b).abs() < 1e-12);
        }
        let mut storage = original;
        let pointer = storage.as_ptr();
        let view = PackedSymmetric::from_slice_mut(3, &mut storage).unwrap();
        let reduced = view.tridiagonal_reduction_in_place().unwrap();
        assert_eq!(reduced.packed_reflectors().as_ptr(), pointer);
    }

    #[test]
    fn complex_reconstruction_and_application() {
        let c = Complex64::new;
        let original = [
            c(4., 0.),
            c(1., -1.),
            c(2., 0.5),
            c(3., 0.),
            c(0.5, -0.25),
            c(2., 0.),
        ];
        let a = PackedHermitian::from_vec(3, original.to_vec()).unwrap();
        let r = a.tridiagonal_reduction().unwrap();
        let q = r.generate_q().unwrap();
        let mut t = [c(0., 0.); 9];
        for i in 0..3 {
            t[i + i * 3] = c(r.diagonal()[i], 0.);
            if i < 2 {
                t[i + (i + 1) * 3] = c(r.off_diagonal()[i], 0.);
                t[i + 1 + i * 3] = c(r.off_diagonal()[i], 0.);
            }
        }
        let mut rec = [c(0., 0.); 9];
        for row in 0..3 {
            for col in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        rec[row + col * 3] += q[row + k * 3] * t[k + l * 3] * q[col + l * 3].conj();
                    }
                }
            }
        }
        let dense = [
            c(4., 0.),
            c(1., -1.),
            c(2., 0.5),
            c(1., 1.),
            c(3., 0.),
            c(0.5, -0.25),
            c(2., -0.5),
            c(0.5, 0.25),
            c(2., 0.),
        ];
        for (a, b) in rec.iter().zip(dense) {
            assert!((*a - b).norm() < 1e-10);
        }
        let mut identity = vec![c(0., 0.); 9];
        for i in 0..3 {
            identity[i + i * 3] = c(1., 0.);
        }
        r.apply_q_in_place(
            ApplySide::Right,
            UnitaryOperation::None,
            3,
            3,
            3,
            &mut identity,
        )
        .unwrap();
        for (a, b) in identity.iter().zip(&q) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn scalar_edges_and_validation() {
        assert!(
            PackedSymmetric::from_vec(0, Vec::<f32>::new())
                .unwrap()
                .tridiagonal_reduction()
                .unwrap()
                .diagonal()
                .is_empty()
        );
        assert_eq!(
            PackedSymmetric::from_vec(1, vec![2f32])
                .unwrap()
                .tridiagonal_reduction()
                .unwrap()
                .diagonal(),
            &[2.]
        );
        let c = Complex32::new;
        assert_eq!(
            PackedHermitian::from_vec(1, vec![c(3., 0.)])
                .unwrap()
                .tridiagonal_reduction()
                .unwrap()
                .diagonal(),
            &[3.]
        );
        let r = PackedSymmetric::from_vec(2, vec![1f64, 0., 2.])
            .unwrap()
            .tridiagonal_reduction()
            .unwrap();
        let mut bad = vec![0.; 4];
        assert!(
            r.apply_q_in_place(
                ApplySide::Left,
                OrthogonalOperation::None,
                3,
                2,
                3,
                &mut bad
            )
            .is_err()
        );
    }
}
