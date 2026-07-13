//! Interoperability between LAPACK triangular storage formats.

use crate::{
    PackedLower, PackedMatrixError, PackedUpper,
    backend::PackedFormatConversion,
    factorization::{check_info, checked_n},
    storage::{PackedStorage, PackedStorageMut},
};
use std::marker::PhantomData;

/// Selects the physically stored triangle of a triangular matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Triangle {
    Lower,
    Upper,
}

impl Triangle {
    fn uplo(self) -> u8 {
        match self {
            Self::Lower => b'L',
            Self::Upper => b'U',
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Lower => "lower",
            Self::Upper => "upper",
        }
    }
}

/// Selects the physical orientation of rectangular full packed (RFP) data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RfpTranspose {
    /// LAPACK's normal RFP representation.
    Normal,
    /// The transpose for real scalars or conjugate transpose for complex scalars.
    Transposed,
}

/// A triangular matrix in column-major full (`TR`) storage.
///
/// `data` has exactly `dimension * dimension` entries. The triangle opposite
/// [`Self::triangle`] contains structural zeros and is not read when converting
/// back to traditional packed storage.
#[derive(Clone, Debug, PartialEq)]
pub struct FullTriangular<T> {
    data: Vec<T>,
    dimension: usize,
    triangle: Triangle,
}

impl<T> FullTriangular<T> {
    pub fn from_vec(
        dimension: usize,
        triangle: Triangle,
        data: Vec<T>,
    ) -> Result<Self, PackedMatrixError> {
        let expected = full_len(dimension)?;
        if data.len() != expected {
            return Err(PackedMatrixError::InvalidLength {
                n: dimension,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            dimension,
            triangle,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub const fn triangle(&self) -> Triangle {
        self.triangle
    }

    /// Returns the column-major leading dimension (`max(1, dimension)`).
    pub fn leading_dimension(&self) -> usize {
        self.dimension.max(1)
    }
}

/// A triangular matrix in LAPACK rectangular full packed (`TF`/RFP) storage.
///
/// RFP is compact like traditional packed storage, but arranges the same
/// `n*(n+1)/2` values as a rectangular column-major array for different LAPACK
/// kernels. It is not directly usable by operations on [`PackedLower`] or
/// [`PackedUpper`]; convert it back first.
#[derive(Clone, Debug)]
pub struct RectangularFullPacked<T, S = Vec<T>> {
    data: S,
    dimension: usize,
    triangle: Triangle,
    transpose: RfpTranspose,
    marker: PhantomData<T>,
}

pub type RectangularFullPackedView<'a, T> = RectangularFullPacked<T, &'a [T]>;
pub type RectangularFullPackedViewMut<'a, T> = RectangularFullPacked<T, &'a mut [T]>;

impl<T, S> RectangularFullPacked<T, S> {
    fn validate_len(dimension: usize, actual: usize) -> Result<(), PackedMatrixError> {
        let expected = packed_len(dimension)?;
        if actual == expected {
            Ok(())
        } else {
            Err(PackedMatrixError::InvalidLength {
                n: dimension,
                expected,
                actual,
            })
        }
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub const fn triangle(&self) -> Triangle {
        self.triangle
    }

    pub const fn transpose(&self) -> RfpTranspose {
        self.transpose
    }

    /// Returns the physical `(rows, columns)` of the column-major RFP rectangle.
    pub fn shape(&self) -> (usize, usize) {
        let normal = if self.dimension % 2 == 0 {
            (self.dimension + 1, self.dimension / 2)
        } else {
            (self.dimension, (self.dimension + 1) / 2)
        };
        match self.transpose {
            RfpTranspose::Normal => normal,
            RfpTranspose::Transposed => (normal.1, normal.0),
        }
    }

    pub fn leading_dimension(&self) -> usize {
        self.shape().0.max(1)
    }
}

impl<T, S: PackedStorage<T>> RectangularFullPacked<T, S> {
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn as_view(&self) -> RectangularFullPackedView<'_, T> {
        RectangularFullPacked {
            data: self.as_slice(),
            dimension: self.dimension,
            triangle: self.triangle,
            transpose: self.transpose,
            marker: PhantomData,
        }
    }
}

impl<T, S: PackedStorageMut<T>> RectangularFullPacked<T, S> {
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    pub fn as_view_mut(&mut self) -> RectangularFullPackedViewMut<'_, T> {
        let dimension = self.dimension;
        let triangle = self.triangle;
        let transpose = self.transpose;
        RectangularFullPacked {
            data: self.as_mut_slice(),
            dimension,
            triangle,
            transpose,
            marker: PhantomData,
        }
    }
}

impl<T> RectangularFullPacked<T> {
    pub fn from_vec(
        dimension: usize,
        triangle: Triangle,
        transpose: RfpTranspose,
        data: Vec<T>,
    ) -> Result<Self, PackedMatrixError> {
        Self::validate_len(dimension, data.len())?;
        Ok(Self {
            data,
            dimension,
            triangle,
            transpose,
            marker: PhantomData,
        })
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<'a, T> RectangularFullPacked<T, &'a [T]> {
    pub fn from_slice(
        dimension: usize,
        triangle: Triangle,
        transpose: RfpTranspose,
        data: &'a [T],
    ) -> Result<Self, PackedMatrixError> {
        Self::validate_len(dimension, data.len())?;
        Ok(Self {
            data,
            dimension,
            triangle,
            transpose,
            marker: PhantomData,
        })
    }
}

impl<'a, T> RectangularFullPacked<T, &'a mut [T]> {
    pub fn from_slice_mut(
        dimension: usize,
        triangle: Triangle,
        transpose: RfpTranspose,
        data: &'a mut [T],
    ) -> Result<Self, PackedMatrixError> {
        Self::validate_len(dimension, data.len())?;
        Ok(Self {
            data,
            dimension,
            triangle,
            transpose,
            marker: PhantomData,
        })
    }
}

fn packed_len(n: usize) -> Result<usize, PackedMatrixError> {
    n.checked_add(1)
        .and_then(|n1| n.checked_mul(n1))
        .map(|len| len / 2)
        .ok_or(PackedMatrixError::DimensionOverflow { n })
}

fn full_len(n: usize) -> Result<usize, PackedMatrixError> {
    n.checked_mul(n)
        .ok_or(PackedMatrixError::DimensionOverflow { n })
}

fn transr<T: PackedFormatConversion>(transpose: RfpTranspose) -> u8 {
    match transpose {
        RfpTranspose::Normal => b'N',
        RfpTranspose::Transposed => T::RFP_TRANSPOSE,
    }
}

fn packed_to_full<T: PackedFormatConversion>(
    n: usize,
    triangle: Triangle,
    packed: &[T],
) -> Result<FullTriangular<T>, PackedMatrixError> {
    let mut data = vec![T::zero(); full_len(n)?];
    let mut info = 0;
    unsafe {
        T::tpttr(
            triangle.uplo(),
            checked_n(n)?,
            packed,
            &mut data,
            checked_n(n.max(1))?,
            &mut info,
        )
    };
    check_info(
        info,
        "traditional packed to full triangular conversion failed",
    )?;
    FullTriangular::from_vec(n, triangle, data)
}

fn full_to_packed<T: PackedFormatConversion>(
    full: &FullTriangular<T>,
    expected: Triangle,
) -> Result<Vec<T>, PackedMatrixError> {
    check_triangle(expected, full.triangle)?;
    let n = full.dimension;
    let mut packed = vec![T::zero(); packed_len(n)?];
    let mut info = 0;
    unsafe {
        T::trttp(
            expected.uplo(),
            checked_n(n)?,
            &full.data,
            checked_n(n.max(1))?,
            &mut packed,
            &mut info,
        )
    };
    check_info(
        info,
        "full triangular to traditional packed conversion failed",
    )?;
    Ok(packed)
}

fn packed_to_rfp_into<T: PackedFormatConversion>(
    n: usize,
    triangle: Triangle,
    transpose: RfpTranspose,
    packed: &[T],
    rfp: &mut [T],
) -> Result<(), PackedMatrixError> {
    let mut info = 0;
    unsafe {
        T::tpttf(
            transr::<T>(transpose),
            triangle.uplo(),
            checked_n(n)?,
            packed,
            rfp,
            &mut info,
        )
    };
    check_info(info, "traditional packed to RFP conversion failed")
}

fn packed_to_rfp<T: PackedFormatConversion>(
    n: usize,
    triangle: Triangle,
    transpose: RfpTranspose,
    packed: &[T],
) -> Result<RectangularFullPacked<T>, PackedMatrixError> {
    let mut data = vec![T::zero(); packed_len(n)?];
    packed_to_rfp_into(n, triangle, transpose, packed, &mut data)?;
    RectangularFullPacked::from_vec(n, triangle, transpose, data)
}

fn rfp_to_packed<T: PackedFormatConversion, S: PackedStorage<T>>(
    rfp: &RectangularFullPacked<T, S>,
    expected: Triangle,
) -> Result<Vec<T>, PackedMatrixError> {
    check_triangle(expected, rfp.triangle)?;
    let mut packed = vec![T::zero(); packed_len(rfp.dimension)?];
    rfp_to_packed_into(
        rfp.dimension,
        expected,
        rfp.transpose,
        rfp.as_slice(),
        &mut packed,
    )?;
    Ok(packed)
}

fn rfp_to_packed_into<T: PackedFormatConversion>(
    n: usize,
    triangle: Triangle,
    transpose: RfpTranspose,
    rfp: &[T],
    packed: &mut [T],
) -> Result<(), PackedMatrixError> {
    let mut info = 0;
    unsafe {
        T::tfttp(
            transr::<T>(transpose),
            triangle.uplo(),
            checked_n(n)?,
            rfp,
            packed,
            &mut info,
        )
    };
    check_info(info, "RFP to traditional packed conversion failed")
}

fn check_triangle(expected: Triangle, actual: Triangle) -> Result<(), PackedMatrixError> {
    if expected == actual {
        Ok(())
    } else {
        Err(PackedMatrixError::TriangleMismatch {
            expected: expected.name(),
            actual: actual.name(),
        })
    }
}

macro_rules! impl_packed_conversions {
    ($packed:ident, $triangle:ident) => {
        impl<T, S> $packed<T, S>
        where
            T: PackedFormatConversion,
            S: PackedStorage<T>,
        {
            /// Copies traditional packed (`TP`) storage to column-major full triangular (`TR`) storage.
            pub fn to_full_triangular(&self) -> Result<FullTriangular<T>, PackedMatrixError> {
                packed_to_full(self.dimension(), Triangle::$triangle, self.as_slice())
            }

            /// Copies traditional packed (`TP`) storage to rectangular full packed (`TF`/RFP) storage.
            pub fn to_rectangular_full_packed(
                &self,
                transpose: RfpTranspose,
            ) -> Result<RectangularFullPacked<T>, PackedMatrixError> {
                packed_to_rfp(
                    self.dimension(),
                    Triangle::$triangle,
                    transpose,
                    self.as_slice(),
                )
            }
        }

        impl<T> $packed<T>
        where
            T: PackedFormatConversion,
        {
            /// Converts full triangular (`TR`) storage back to traditional packed (`TP`) storage.
            pub fn from_full_triangular(
                full: &FullTriangular<T>,
            ) -> Result<Self, PackedMatrixError> {
                Self::from_vec(full.dimension(), full_to_packed(full, Triangle::$triangle)?)
            }

            /// Converts rectangular full packed (`TF`/RFP) storage back to traditional packed (`TP`) storage.
            pub fn from_rectangular_full_packed<S: PackedStorage<T>>(
                rfp: &RectangularFullPacked<T, S>,
            ) -> Result<Self, PackedMatrixError> {
                Self::from_vec(rfp.dimension(), rfp_to_packed(rfp, Triangle::$triangle)?)
            }

            /// Consumes traditional packed storage, reusing its allocation for the RFP result.
            pub fn into_rectangular_full_packed(
                self,
                transpose: RfpTranspose,
            ) -> Result<RectangularFullPacked<T>, PackedMatrixError> {
                let n = self.dimension();
                let mut data = self.into_vec();
                let packed = data.clone();
                packed_to_rfp_into(n, Triangle::$triangle, transpose, &packed, &mut data)?;
                RectangularFullPacked::from_vec(n, Triangle::$triangle, transpose, data)
            }
        }
    };
}

impl_packed_conversions!(PackedLower, Lower);
impl_packed_conversions!(PackedUpper, Upper);

impl<T, S> RectangularFullPacked<T, S>
where
    T: PackedFormatConversion,
    S: PackedStorage<T>,
{
    pub fn to_packed_lower(&self) -> Result<PackedLower<T>, PackedMatrixError> {
        PackedLower::from_rectangular_full_packed(self)
    }

    pub fn to_packed_upper(&self) -> Result<PackedUpper<T>, PackedMatrixError> {
        PackedUpper::from_rectangular_full_packed(self)
    }
}

impl<T> RectangularFullPacked<T>
where
    T: PackedFormatConversion,
{
    /// Consumes lower RFP storage, reusing its allocation for traditional packed storage.
    pub fn into_packed_lower(self) -> Result<PackedLower<T>, PackedMatrixError> {
        check_triangle(Triangle::Lower, self.triangle)?;
        let n = self.dimension;
        let transpose = self.transpose;
        let mut data = self.data;
        let rfp = data.clone();
        rfp_to_packed_into(n, Triangle::Lower, transpose, &rfp, &mut data)?;
        PackedLower::from_vec(n, data)
    }

    /// Consumes upper RFP storage, reusing its allocation for traditional packed storage.
    pub fn into_packed_upper(self) -> Result<PackedUpper<T>, PackedMatrixError> {
        check_triangle(Triangle::Upper, self.triangle)?;
        let n = self.dimension;
        let transpose = self.transpose;
        let mut data = self.data;
        let rfp = data.clone();
        rfp_to_packed_into(n, Triangle::Upper, transpose, &rfp, &mut data)?;
        PackedUpper::from_vec(n, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    macro_rules! round_trips {
        ($scalar:ty, $value:expr) => {{
            for n in [0usize, 1, 4, 5] {
                let len = packed_len(n).unwrap();
                let data: Vec<$scalar> = (0..len).map($value).collect();
                for transpose in [RfpTranspose::Normal, RfpTranspose::Transposed] {
                    let lower = PackedLower::from_vec(n, data.clone()).unwrap();
                    let full = lower.to_full_triangular().unwrap();
                    assert_eq!(
                        PackedLower::from_full_triangular(&full).unwrap().as_slice(),
                        data
                    );
                    let rfp = lower.to_rectangular_full_packed(transpose).unwrap();
                    assert_eq!(rfp.to_packed_lower().unwrap().as_slice(), data);

                    let upper = PackedUpper::from_vec(n, data.clone()).unwrap();
                    let full = upper.to_full_triangular().unwrap();
                    assert_eq!(
                        PackedUpper::from_full_triangular(&full).unwrap().as_slice(),
                        data
                    );
                    let pointer = upper.as_slice().as_ptr();
                    let rfp = upper.into_rectangular_full_packed(transpose).unwrap();
                    assert_eq!(rfp.as_slice().as_ptr(), pointer);
                    let pointer = rfp.as_slice().as_ptr();
                    let packed = rfp.into_packed_upper().unwrap();
                    assert_eq!(packed.as_slice().as_ptr(), pointer);
                    assert_eq!(packed.as_slice(), data);
                }
            }
        }};
    }

    #[test]
    fn every_scalar_odd_even_orientation_and_transform() {
        round_trips!(f32, |i| i as f32 + 1.0);
        round_trips!(f64, |i| i as f64 + 1.0);
        round_trips!(Complex32, |i| Complex32::new(i as f32 + 1.0, i as f32));
        round_trips!(Complex64, |i| Complex64::new(i as f64 + 1.0, i as f64));
    }

    #[test]
    fn metadata_views_structural_zeros_and_validation() {
        let lower = PackedLower::from_vec(2, vec![1f64, 2., 3.]).unwrap();
        let full = lower.to_full_triangular().unwrap();
        assert_eq!(full.as_slice(), &[1., 2., 0., 3.]);
        assert_eq!(full.leading_dimension(), 2);
        assert!(PackedUpper::from_full_triangular(&full).is_err());

        let mut data = vec![1f64, 2., 3., 4., 5., 6.];
        let mut view = RectangularFullPackedViewMut::from_slice_mut(
            3,
            Triangle::Lower,
            RfpTranspose::Transposed,
            &mut data,
        )
        .unwrap();
        assert_eq!(view.shape(), (2, 3));
        assert_eq!(view.leading_dimension(), 2);
        assert_eq!(view.as_view().triangle(), Triangle::Lower);
        assert_eq!(view.as_mut_slice().len(), 6);
        assert!(view.to_packed_upper().is_err());

        assert!(FullTriangular::<f32>::from_vec(2, Triangle::Lower, vec![0.; 3]).is_err());
        assert!(
            RectangularFullPacked::<f32>::from_vec(
                2,
                Triangle::Lower,
                RfpTranspose::Normal,
                vec![0.; 2],
            )
            .is_err()
        );
    }
}
