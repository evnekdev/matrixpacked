//! Options and result types shared by packed triangular BLAS/LAPACK operations.

/// Matrix operation applied by a triangular multiply or solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Transpose {
    /// Use `A` directly.
    #[default]
    None,
    /// Use the transpose `A^T`.
    Transpose,
    /// Use the conjugate transpose `A^H` (`A^T` for real scalars).
    ConjugateTranspose,
}

impl Transpose {
    pub(crate) const fn as_lapack(self) -> u8 {
        match self {
            Self::None => b'N',
            Self::Transpose => b'T',
            Self::ConjugateTranspose => b'C',
        }
    }
}

/// Whether diagonal entries are read from packed storage or treated as one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Diagonal {
    #[default]
    NonUnit,
    Unit,
}

impl Diagonal {
    pub(crate) const fn as_lapack(self) -> u8 {
        match self {
            Self::NonUnit => b'N',
            Self::Unit => b'U',
        }
    }
}

/// Norms supported by LAPACK's `xLANTP` packed triangular norm routine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixNorm {
    MaxAbs,
    One,
    Infinity,
    Frobenius,
}

impl MatrixNorm {
    pub(crate) const fn as_lapack(self) -> u8 {
        match self {
            Self::MaxAbs => b'M',
            Self::One => b'1',
            Self::Infinity => b'I',
            Self::Frobenius => b'F',
        }
    }
}

/// Norm used for a reciprocal condition-number estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditionNorm {
    One,
    Infinity,
}

impl ConditionNorm {
    pub(crate) const fn as_lapack(self) -> u8 {
        match self {
            Self::One => b'1',
            Self::Infinity => b'I',
        }
    }
}

/// Error estimates returned by LAPACK's `xTPRFS` iterative refinement.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinementReport<R> {
    /// Estimated forward error bound for each right-hand side.
    pub forward_error: Vec<R>,
    /// Componentwise relative backward error for each right-hand side.
    pub backward_error: Vec<R>,
}

pub(crate) fn check_strided_vector(
    n: usize,
    len: usize,
    increment: i32,
) -> Result<(), crate::PackedMatrixError> {
    let stride = increment
        .checked_abs()
        .ok_or(crate::PackedMatrixError::InvalidIncrement { increment })? as usize;
    if stride == 0 {
        return Err(crate::PackedMatrixError::InvalidIncrement { increment });
    }
    let expected = if n == 0 {
        0
    } else {
        1 + (n - 1)
            .checked_mul(stride)
            .ok_or(crate::PackedMatrixError::DimensionOverflow { n })?
    };
    if len < expected {
        return Err(crate::PackedMatrixError::InvalidVectorLength {
            expected,
            actual: len,
        });
    }
    Ok(())
}

macro_rules! impl_triangular_packed_ops {
    ($name:ident, $uplo:expr, $label:literal) => {
        impl<T, S> $name<T, S>
        where
            T: crate::backend::TriangularPackedBackend,
            S: crate::storage::PackedStorage<T>,
        {
            /// Computes `x := op(A)*x` in place using BLAS `xTPMV` without copying `A` or `x`.
            pub fn mul_vector_strided_in_place(
                &self,
                x: &mut [T],
                increment: i32,
                op: crate::Transpose,
                diagonal: crate::Diagonal,
            ) -> Result<(), crate::PackedMatrixError> {
                crate::triangular::check_strided_vector(self.n, x.len(), increment)?;
                unsafe {
                    T::tpmv(
                        $uplo,
                        op.as_lapack(),
                        diagonal.as_lapack(),
                        crate::factorization::checked_n(self.n)?,
                        self.as_slice(),
                        x,
                        increment,
                    )
                };
                Ok(())
            }
            pub fn mul_vector_op_in_place(
                &self,
                x: &mut [T],
                op: crate::Transpose,
                diagonal: crate::Diagonal,
            ) -> Result<(), crate::PackedMatrixError> {
                self.mul_vector_strided_in_place(x, 1, op, diagonal)
            }
            pub fn mul_vector_in_place(&self, x: &mut [T]) -> Result<(), crate::PackedMatrixError> {
                self.mul_vector_op_in_place(x, crate::Transpose::None, crate::Diagonal::NonUnit)
            }
            pub fn mul_vector_op(
                &self,
                x: &[T],
                op: crate::Transpose,
                diagonal: crate::Diagonal,
            ) -> Result<Vec<T>, crate::PackedMatrixError> {
                let mut y = x.to_vec();
                self.mul_vector_op_in_place(&mut y, op, diagonal)?;
                Ok(y)
            }
            pub fn mul_vector(&self, x: &[T]) -> Result<Vec<T>, crate::PackedMatrixError> {
                self.mul_vector_op(x, crate::Transpose::None, crate::Diagonal::NonUnit)
            }

            /// Solves `op(A)*x=b` in place with the single-vector BLAS `xTPSV` routine.
            pub fn solve_vector_blas_strided_in_place(
                &self,
                x: &mut [T],
                increment: i32,
                op: crate::Transpose,
                diagonal: crate::Diagonal,
            ) -> Result<(), crate::PackedMatrixError> {
                crate::triangular::check_strided_vector(self.n, x.len(), increment)?;
                unsafe {
                    T::tpsv(
                        $uplo,
                        op.as_lapack(),
                        diagonal.as_lapack(),
                        crate::factorization::checked_n(self.n)?,
                        self.as_slice(),
                        x,
                        increment,
                    )
                };
                Ok(())
            }
            pub fn solve_vector_blas_in_place(
                &self,
                x: &mut [T],
                op: crate::Transpose,
                diagonal: crate::Diagonal,
            ) -> Result<(), crate::PackedMatrixError> {
                self.solve_vector_blas_strided_in_place(x, 1, op, diagonal)
            }
            pub fn solve_vector_blas(
                &self,
                b: &[T],
                op: crate::Transpose,
                diagonal: crate::Diagonal,
            ) -> Result<Vec<T>, crate::PackedMatrixError> {
                let mut x = b.to_vec();
                self.solve_vector_blas_in_place(&mut x, op, diagonal)?;
                Ok(x)
            }

            /// Solves one or more column-major right-hand sides with LAPACK `xTPTRS`.
            pub fn solve_many_in_place(
                &self,
                b: &mut [T],
                nrhs: usize,
                op: crate::Transpose,
                diagonal: crate::Diagonal,
            ) -> Result<(), crate::PackedMatrixError> {
                crate::factorization::check_rhs_many(self.n, nrhs, b)?;
                let n = crate::factorization::checked_n(self.n)?;
                let mut info = 0;
                unsafe {
                    T::tptrs(
                        $uplo,
                        op.as_lapack(),
                        diagonal.as_lapack(),
                        n,
                        crate::factorization::checked_n(nrhs)?,
                        self.as_slice(),
                        b,
                        n.max(1),
                        &mut info,
                    )
                };
                crate::factorization::check_info(
                    info,
                    concat!($label, " packed triangular solve failed"),
                )
            }
            pub fn solve_vector_in_place(
                &self,
                b: &mut [T],
            ) -> Result<(), crate::PackedMatrixError> {
                self.solve_many_in_place(b, 1, crate::Transpose::None, crate::Diagonal::NonUnit)
            }
            pub fn solve_vector(&self, b: &[T]) -> Result<Vec<T>, crate::PackedMatrixError> {
                let mut x = b.to_vec();
                self.solve_vector_in_place(&mut x)?;
                Ok(x)
            }

            /// Estimates `1 / cond(A)` with LAPACK `xTPCON` without forming `A^-1`.
            pub fn rcond(
                &self,
                norm: crate::ConditionNorm,
                diagonal: crate::Diagonal,
            ) -> Result<T::Real, crate::PackedMatrixError> {
                let n = crate::factorization::checked_n(self.n)?;
                let mut r = <T::Real as num_traits::Zero>::zero();
                let mut work = vec![
                    T::zero();
                    crate::factorization::checked_workspace_len(
                        self.n,
                        if T::IS_COMPLEX { 2 } else { 3 }
                    )?
                ];
                let mut rw = vec![
                    <T::Real as num_traits::Zero>::zero();
                    if T::IS_COMPLEX { self.n } else { 0 }
                ];
                let mut iw = vec![0; if T::IS_COMPLEX { 0 } else { self.n }];
                let mut info = 0;
                unsafe {
                    T::tpcon(
                        norm.as_lapack(),
                        $uplo,
                        diagonal.as_lapack(),
                        n,
                        self.as_slice(),
                        &mut r,
                        &mut work,
                        &mut rw,
                        &mut iw,
                        &mut info,
                    )
                };
                crate::factorization::check_info(
                    info,
                    "packed triangular condition estimate failed",
                )?;
                Ok(r)
            }
            /// Long-form alias for [`Self::rcond`].
            pub fn reciprocal_condition_number(
                &self,
                norm: crate::ConditionNorm,
                diagonal: crate::Diagonal,
            ) -> Result<T::Real, crate::PackedMatrixError> {
                self.rcond(norm, diagonal)
            }

            /// Computes a packed triangular matrix norm with LAPACK `xLANTP`.
            pub fn matrix_norm(
                &self,
                norm: crate::MatrixNorm,
                diagonal: crate::Diagonal,
            ) -> Result<T::Real, crate::PackedMatrixError> {
                let mut work = vec![<T::Real as num_traits::Zero>::zero(); self.n];
                Ok(unsafe {
                    T::lantp(
                        norm.as_lapack(),
                        $uplo,
                        diagonal.as_lapack(),
                        crate::factorization::checked_n(self.n)?,
                        self.as_slice(),
                        &mut work,
                    )
                })
            }

            /// Refines column-major solutions in place and returns LAPACK forward/backward errors.
            pub fn refine_many_in_place(
                &self,
                b: &[T],
                x: &mut [T],
                nrhs: usize,
                op: crate::Transpose,
                diagonal: crate::Diagonal,
            ) -> Result<crate::RefinementReport<T::Real>, crate::PackedMatrixError> {
                crate::factorization::check_rhs_many(self.n, nrhs, b)?;
                crate::factorization::check_rhs_many(self.n, nrhs, x)?;
                let n = crate::factorization::checked_n(self.n)?;
                let mut ferr = vec![<T::Real as num_traits::Zero>::zero(); nrhs];
                let mut berr = ferr.clone();
                let mut work = vec![
                    T::zero();
                    crate::factorization::checked_workspace_len(
                        self.n,
                        if T::IS_COMPLEX { 2 } else { 3 }
                    )?
                ];
                let mut rw = vec![
                    <T::Real as num_traits::Zero>::zero();
                    if T::IS_COMPLEX { self.n } else { 0 }
                ];
                let mut iw = vec![0; if T::IS_COMPLEX { 0 } else { self.n }];
                let mut info = 0;
                unsafe {
                    T::tprfs(
                        $uplo,
                        op.as_lapack(),
                        diagonal.as_lapack(),
                        n,
                        crate::factorization::checked_n(nrhs)?,
                        self.as_slice(),
                        b,
                        n.max(1),
                        x,
                        n.max(1),
                        &mut ferr,
                        &mut berr,
                        &mut work,
                        &mut rw,
                        &mut iw,
                        &mut info,
                    )
                };
                crate::factorization::check_info(
                    info,
                    "packed triangular iterative refinement failed",
                )?;
                Ok(crate::RefinementReport {
                    forward_error: ferr,
                    backward_error: berr,
                })
            }
        }
        impl<T, S> $name<T, S>
        where
            T: crate::backend::TriangularPackedBackend,
            S: crate::storage::PackedStorageMut<T>,
        {
            /// Inverts the packed triangular matrix in place using LAPACK `xTPTRI`.
            pub fn inverse_in_place_with_diagonal(
                &mut self,
                diagonal: crate::Diagonal,
            ) -> Result<(), crate::PackedMatrixError> {
                let mut info = 0;
                unsafe {
                    T::tptri(
                        $uplo,
                        diagonal.as_lapack(),
                        crate::factorization::checked_n(self.n)?,
                        self.as_mut_slice(),
                        &mut info,
                    )
                };
                crate::factorization::check_info(
                    info,
                    concat!($label, " packed triangular inverse failed"),
                )
            }
            pub fn inverse_in_place(&mut self) -> Result<(), crate::PackedMatrixError> {
                self.inverse_in_place_with_diagonal(crate::Diagonal::NonUnit)
            }
        }
    };
}
pub(crate) use impl_triangular_packed_ops;
