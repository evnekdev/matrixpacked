//! Shared elementwise arithmetic implementations.

macro_rules! impl_packed_ring_ops {
    ($name:ident) => {
        impl<T, L, R> std::ops::Add<&$name<T, R>> for &$name<T, L>
        where
            T: crate::LapackScalar,
            L: crate::storage::PackedStorage<T>,
            R: crate::storage::PackedStorage<T>,
        {
            type Output = $name<T>;
            fn add(self, rhs: &$name<T, R>) -> Self::Output {
                assert_eq!(
                    self.dimension(),
                    rhs.dimension(),
                    "matrix dimensions must match"
                );
                $name::from_vec(
                    self.dimension(),
                    self.as_slice()
                        .iter()
                        .zip(rhs.as_slice())
                        .map(|(&a, &b)| a + b)
                        .collect(),
                )
                .expect("validated packed length")
            }
        }
        impl<T, S, R> std::ops::AddAssign<&$name<T, R>> for $name<T, S>
        where
            T: crate::LapackScalar,
            S: crate::storage::PackedStorageMut<T>,
            R: crate::storage::PackedStorage<T>,
        {
            fn add_assign(&mut self, rhs: &$name<T, R>) {
                assert_eq!(
                    self.dimension(),
                    rhs.dimension(),
                    "matrix dimensions must match"
                );
                for (a, &b) in self.as_mut_slice().iter_mut().zip(rhs.as_slice()) {
                    *a += b;
                }
            }
        }
        impl<T, L, R> std::ops::Sub<&$name<T, R>> for &$name<T, L>
        where
            T: crate::LapackScalar,
            L: crate::storage::PackedStorage<T>,
            R: crate::storage::PackedStorage<T>,
        {
            type Output = $name<T>;
            fn sub(self, rhs: &$name<T, R>) -> Self::Output {
                assert_eq!(
                    self.dimension(),
                    rhs.dimension(),
                    "matrix dimensions must match"
                );
                $name::from_vec(
                    self.dimension(),
                    self.as_slice()
                        .iter()
                        .zip(rhs.as_slice())
                        .map(|(&a, &b)| a - b)
                        .collect(),
                )
                .expect("validated packed length")
            }
        }
        impl<T, S, R> std::ops::SubAssign<&$name<T, R>> for $name<T, S>
        where
            T: crate::LapackScalar,
            S: crate::storage::PackedStorageMut<T>,
            R: crate::storage::PackedStorage<T>,
        {
            fn sub_assign(&mut self, rhs: &$name<T, R>) {
                assert_eq!(
                    self.dimension(),
                    rhs.dimension(),
                    "matrix dimensions must match"
                );
                for (a, &b) in self.as_mut_slice().iter_mut().zip(rhs.as_slice()) {
                    *a -= b;
                }
            }
        }
        impl<T, S> std::ops::Neg for &$name<T, S>
        where
            T: crate::LapackScalar,
            S: crate::storage::PackedStorage<T>,
        {
            type Output = $name<T>;
            fn neg(self) -> Self::Output {
                $name::from_vec(
                    self.dimension(),
                    self.as_slice().iter().map(|&x| -x).collect(),
                )
                .expect("validated packed length")
            }
        }
        impl<T, S> std::ops::Mul<T> for &$name<T, S>
        where
            T: crate::LapackScalar,
            S: crate::storage::PackedStorage<T>,
        {
            type Output = $name<T>;
            fn mul(self, rhs: T) -> Self::Output {
                $name::from_vec(
                    self.dimension(),
                    self.as_slice().iter().map(|&x| x * rhs).collect(),
                )
                .expect("validated packed length")
            }
        }
        impl<T, S> std::ops::Div<T> for &$name<T, S>
        where
            T: crate::LapackScalar,
            S: crate::storage::PackedStorage<T>,
        {
            type Output = $name<T>;
            fn div(self, rhs: T) -> Self::Output {
                $name::from_vec(
                    self.dimension(),
                    self.as_slice().iter().map(|&x| x / rhs).collect(),
                )
                .expect("validated packed length")
            }
        }
        impl<T, S> std::ops::MulAssign<T> for $name<T, S>
        where
            T: crate::LapackScalar,
            S: crate::storage::PackedStorageMut<T>,
        {
            fn mul_assign(&mut self, rhs: T) {
                for x in self.as_mut_slice() {
                    *x *= rhs;
                }
            }
        }
        impl<T, S> std::ops::DivAssign<T> for $name<T, S>
        where
            T: crate::LapackScalar,
            S: crate::storage::PackedStorageMut<T>,
        {
            fn div_assign(&mut self, rhs: T) {
                for x in self.as_mut_slice() {
                    *x /= rhs;
                }
            }
        }
        impl<T, S> $name<T, S>
        where
            T: crate::LapackScalar,
            S: crate::storage::PackedStorage<T>,
        {
            pub fn component_mul<R: crate::storage::PackedStorage<T>>(
                &self,
                rhs: &$name<T, R>,
            ) -> Result<$name<T>, crate::PackedMatrixError> {
                if self.dimension() != rhs.dimension() {
                    return Err(crate::PackedMatrixError::DimensionMismatch {
                        left: self.dimension(),
                        right: rhs.dimension(),
                    });
                }
                $name::from_vec(
                    self.dimension(),
                    self.as_slice()
                        .iter()
                        .zip(rhs.as_slice())
                        .map(|(&a, &b)| a * b)
                        .collect(),
                )
            }
            pub fn component_div<R: crate::storage::PackedStorage<T>>(
                &self,
                rhs: &$name<T, R>,
            ) -> Result<$name<T>, crate::PackedMatrixError> {
                if self.dimension() != rhs.dimension() {
                    return Err(crate::PackedMatrixError::DimensionMismatch {
                        left: self.dimension(),
                        right: rhs.dimension(),
                    });
                }
                $name::from_vec(
                    self.dimension(),
                    self.as_slice()
                        .iter()
                        .zip(rhs.as_slice())
                        .map(|(&a, &b)| a / b)
                        .collect(),
                )
            }
            pub fn stored_norm_squared(&self) -> T::Real {
                self.as_slice()
                    .iter()
                    .fold(<T::Real as num_traits::Zero>::zero(), |acc, &x| {
                        acc + x.abs_squared()
                    })
            }
        }
    };
}
pub(crate) use impl_packed_ring_ops;
