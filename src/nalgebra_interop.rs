//! Optional conversions between matrixpacked storage and nalgebra.
//!
//! Extraction constructors deliberately ignore the opposite triangle. Strict
//! constructors validate it using
//! `|a-b| <= absolute + relative * max(|a|, |b|)`. SPD/HPD strict conversion
//! additionally uses nalgebra Cholesky to prove positive definiteness. Every
//! conversion allocates owned storage; there are no zero-copy nalgebra views.
//!
//! # Examples
//!
//! ```
//! use matrixpacked::{ConversionTolerance, PackedSymmetric};
//! use nalgebra::DMatrix;
//!
//! let nonsymmetric = DMatrix::from_row_slice(2, 2, &[1.0_f64, 9.0, 2.0, 3.0]);
//! let extracted = PackedSymmetric::from_lower_triangle(&nonsymmetric)?;
//! assert_eq!(extracted.as_slice(), &[1.0, 2.0, 3.0]);
//! assert!(PackedSymmetric::try_from_dmatrix(
//!     &nonsymmetric,
//!     ConversionTolerance::new(0.0, 0.0),
//! ).is_err());
//! # Ok::<(), matrixpacked::PackedMatrixError>(())
//! ```

use crate::{
    FullTriangular, PackedHermitian, PackedLower, PackedMatrixError, PackedSPD, PackedSymmetric,
    PackedUpper, Triangle, backend::PackedFormatConversion, storage::PackedStorage,
};
use nalgebra::{ComplexField, DMatrix, Scalar, linalg::Cholesky};
use num_complex::{Complex32, Complex64};
use num_traits::{Float, Zero};

/// Absolute and relative thresholds for validated nalgebra conversions.
///
/// Two scalar values `a` and `b` are accepted when
/// `|a - b| <= absolute + relative * max(|a|, |b|)`. Complex magnitudes are
/// used for complex scalars. Both components must be finite and nonnegative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConversionTolerance<R> {
    /// Absolute error allowed regardless of operand magnitude.
    pub absolute: R,
    /// Relative error multiplied by the larger operand magnitude.
    pub relative: R,
}

impl<R> ConversionTolerance<R> {
    /// Creates a tolerance. Validation occurs when it is used by a conversion.
    pub const fn new(absolute: R, relative: R) -> Self {
        Self { absolute, relative }
    }
}

/// Supplies a conservative default conversion tolerance for a real scalar type.
///
/// Defaults use zero absolute tolerance and eight machine epsilons of relative
/// tolerance. Callers should choose explicit tolerances when their data has a
/// known scale or error model.
pub trait DefaultConversionTolerance: Sized {
    /// Returns zero absolute tolerance and eight machine epsilons of relative tolerance.
    fn default_conversion_tolerance() -> ConversionTolerance<Self>;
}

macro_rules! impl_default_conversion_tolerance {
    ($type:ty) => {
        impl DefaultConversionTolerance for $type {
            fn default_conversion_tolerance() -> ConversionTolerance<Self> {
                ConversionTolerance::new(0.0, 8.0 * <$type>::EPSILON)
            }
        }

        impl Default for ConversionTolerance<$type> {
            fn default() -> Self {
                <$type>::default_conversion_tolerance()
            }
        }
    };
}

impl_default_conversion_tolerance!(f32);
impl_default_conversion_tolerance!(f64);

impl<T> FullTriangular<T>
where
    T: Scalar,
{
    /// Clones this full column-major buffer into a nalgebra dynamic matrix.
    ///
    /// The result allocates `n * n` column-major entries and clones every scalar;
    /// the source remains available. No structure is validated or triangle ignored,
    /// and no native backend is called. Available for any nalgebra [`Scalar`].
    pub fn to_dmatrix(&self) -> DMatrix<T> {
        DMatrix::from_vec(self.dimension(), self.dimension(), self.as_slice().to_vec())
    }

    /// Moves this full column-major buffer into a nalgebra dynamic matrix.
    ///
    /// Unlike [`Self::to_dmatrix`], this consumes `self`, reuses its `n * n`
    /// column-major allocation, and does not clone, validate, discard entries, or
    /// call a native backend. Available for any nalgebra [`Scalar`].
    pub fn into_dmatrix(self) -> DMatrix<T> {
        let dimension = self.dimension();
        DMatrix::from_vec(dimension, dimension, self.into_vec())
    }
}

impl<T> FullTriangular<T>
where
    T: Scalar + Zero,
{
    /// Copies a square nalgebra matrix into full triangular storage.
    ///
    /// Nalgebra and `FullTriangular` both use column-major storage, so the full
    /// buffer is copied directly. Entries outside `triangle` are then replaced
    /// by structural zeros to preserve the [`FullTriangular`] invariant.
    /// The conversion borrows `matrix`, clones all `n * n` column-major entries,
    /// then ignores (zeros) the opposite triangle. It validates only square shape
    /// and returns [`PackedMatrixError::NonSquareMatrix`] otherwise. It accepts any
    /// nalgebra [`Scalar`] with [`Zero`] and does not call a native backend.
    pub fn try_from_dmatrix(
        matrix: &DMatrix<T>,
        triangle: Triangle,
    ) -> Result<Self, PackedMatrixError> {
        let (rows, columns) = matrix.shape();
        if rows != columns {
            return Err(PackedMatrixError::NonSquareMatrix { rows, columns });
        }

        let mut data = matrix.as_slice().to_vec();
        for column in 0..columns {
            for row in 0..rows {
                let is_structural_zero = match triangle {
                    Triangle::Lower => row < column,
                    Triangle::Upper => row > column,
                };
                if is_structural_zero {
                    data[column * rows + row] = T::zero();
                }
            }
        }

        Self::from_vec(rows, triangle, data)
    }
}

impl<T> From<FullTriangular<T>> for DMatrix<T>
where
    T: Scalar,
{
    fn from(matrix: FullTriangular<T>) -> Self {
        matrix.into_dmatrix()
    }
}

macro_rules! impl_packed_to_dmatrix {
    ($packed:ident) => {
        #[allow(private_bounds)]
        impl<T, S> $packed<T, S>
        where
            T: PackedFormatConversion + Scalar,
            S: PackedStorage<T>,
        {
            /// Converts traditional packed (`TP`) storage to an owned nalgebra matrix.
            ///
            /// LAPACK first expands the selected triangle to a full triangular (`TR`)
            /// buffer with structural zeros in the opposite triangle. The compatible
            /// column-major buffer is then moved into nalgebra. This borrows rather
            /// than consumes the packed source, allocates `n * n` entries, and copies
            /// the packed scalars during expansion. The opposite triangle becomes
            /// structural zero; no input structure is validated. LAPACK failures are
            /// returned as [`PackedMatrixError`]. This supports matrixpacked's four
            /// LAPACK scalar types and requires a linked LAPACK provider at final link.
            pub fn to_dmatrix(&self) -> Result<DMatrix<T>, PackedMatrixError> {
                Ok(self.to_full_triangular()?.into_dmatrix())
            }
        }
    };
}

impl_packed_to_dmatrix!(PackedLower);
impl_packed_to_dmatrix!(PackedUpper);

#[allow(private_bounds)]
impl<T> PackedLower<T>
where
    T: PackedFormatConversion + Scalar + Zero,
{
    /// Extracts the lower triangle of a square nalgebra matrix into owned packed storage.
    ///
    /// Values above the diagonal are intentionally discarded, not validated. LAPACK
    /// converts the resulting full triangular (`TR`) buffer to traditional packed
    /// (`TP`) column-packed storage. The borrowed input is not consumed or changed;
    /// scalars are copied into new full and packed allocations. A rectangular input
    /// returns [`PackedMatrixError::NonSquareMatrix`], and LAPACK argument failures
    /// are propagated. This supports the four LAPACK scalar types and requires a
    /// linked LAPACK provider.
    pub fn from_lower_triangle(matrix: &DMatrix<T>) -> Result<Self, PackedMatrixError> {
        let full = FullTriangular::try_from_dmatrix(matrix, Triangle::Lower)?;
        Self::from_full_triangular(&full)
    }
}

#[allow(private_bounds)]
impl<T> PackedUpper<T>
where
    T: PackedFormatConversion + Scalar + Zero,
{
    /// Extracts the upper triangle of a square nalgebra matrix into owned packed storage.
    ///
    /// Values below the diagonal are intentionally discarded, not validated. LAPACK
    /// converts the resulting full triangular (`TR`) buffer to traditional packed
    /// (`TP`) column-packed storage. The borrowed input is not consumed or changed;
    /// scalars are copied into new full and packed allocations. A rectangular input
    /// returns [`PackedMatrixError::NonSquareMatrix`], and LAPACK argument failures
    /// are propagated. This supports the four LAPACK scalar types and requires a
    /// linked LAPACK provider.
    pub fn from_upper_triangle(matrix: &DMatrix<T>) -> Result<Self, PackedMatrixError> {
        let full = FullTriangular::try_from_dmatrix(matrix, Triangle::Upper)?;
        Self::from_full_triangular(&full)
    }
}

trait StructuredScalar: Scalar + Copy + Zero {
    fn conjugate(self) -> Self;
    fn hermitian_diagonal(self) -> Self;
}

macro_rules! impl_real_structured_scalar {
    ($type:ty) => {
        impl StructuredScalar for $type {
            fn conjugate(self) -> Self {
                self
            }

            fn hermitian_diagonal(self) -> Self {
                self
            }
        }
    };
}

impl_real_structured_scalar!(f32);
impl_real_structured_scalar!(f64);

macro_rules! impl_complex_structured_scalar {
    ($type:ty) => {
        impl StructuredScalar for $type {
            fn conjugate(self) -> Self {
                self.conj()
            }

            fn hermitian_diagonal(self) -> Self {
                Self::new(self.re, 0.0)
            }
        }
    };
}

impl_complex_structured_scalar!(Complex32);
impl_complex_structured_scalar!(Complex64);

trait ValidationScalar:
    StructuredScalar
    + crate::LapackScalar
    + ComplexField<RealField = <Self as crate::LapackScalar>::Real>
where
    <Self as crate::LapackScalar>::Real: Float,
{
    const IS_COMPLEX: bool;
}

macro_rules! impl_validation_scalar {
    ($type:ty, $is_complex:expr) => {
        impl ValidationScalar for $type {
            const IS_COMPLEX: bool = $is_complex;
        }
    };
}

impl_validation_scalar!(f32, false);
impl_validation_scalar!(f64, false);
impl_validation_scalar!(Complex32, true);
impl_validation_scalar!(Complex64, true);

fn validate_tolerance<R: Float>(
    tolerance: ConversionTolerance<R>,
) -> Result<(), PackedMatrixError> {
    for (component, value) in [
        ("absolute", tolerance.absolute),
        ("relative", tolerance.relative),
    ] {
        if !value.is_finite() {
            return Err(PackedMatrixError::InvalidTolerance {
                component,
                reason: "must be finite",
            });
        }
        if value < R::zero() {
            return Err(PackedMatrixError::InvalidTolerance {
                component,
                reason: "must be nonnegative",
            });
        }
    }
    Ok(())
}

fn approximately_equal<T: ValidationScalar>(
    left: T,
    right: T,
    tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
) -> bool
where
    <T as crate::LapackScalar>::Real: Float,
{
    let difference = (left - right).modulus();
    let left_magnitude = left.modulus();
    let right_magnitude = right.modulus();
    let scale = if left_magnitude > right_magnitude {
        left_magnitude
    } else {
        right_magnitude
    };
    difference <= tolerance.absolute + tolerance.relative * scale
}

fn approximately_real<T: ValidationScalar>(
    value: T,
    tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
) -> bool
where
    <T as crate::LapackScalar>::Real: Float,
{
    let imaginary = value.imaginary().abs();
    imaginary <= tolerance.absolute + tolerance.relative * value.modulus()
}

fn square_dimension<T>(matrix: &DMatrix<T>) -> Result<usize, PackedMatrixError> {
    let (rows, columns) = matrix.shape();
    if rows != columns {
        return Err(PackedMatrixError::NonSquareMatrix { rows, columns });
    }
    Ok(rows)
}

fn validate_triangle<T: ValidationScalar>(
    matrix: &DMatrix<T>,
    triangle: Triangle,
    tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
) -> Result<usize, PackedMatrixError>
where
    <T as crate::LapackScalar>::Real: Float,
{
    let dimension = square_dimension(matrix)?;
    validate_tolerance(tolerance)?;
    for column in 0..dimension {
        for row in 0..dimension {
            let is_opposite = match triangle {
                Triangle::Lower => row < column,
                Triangle::Upper => row > column,
            };
            if is_opposite && !approximately_equal(matrix[(row, column)], T::zero(), tolerance) {
                return Err(PackedMatrixError::NotTriangular {
                    triangle: match triangle {
                        Triangle::Lower => "lower",
                        Triangle::Upper => "upper",
                    },
                    row,
                    column,
                });
            }
        }
    }
    Ok(dimension)
}

fn pack_triangle<T: ValidationScalar>(matrix: &DMatrix<T>, triangle: Triangle) -> Vec<T>
where
    <T as crate::LapackScalar>::Real: Float,
{
    let dimension = matrix.nrows();
    let mut packed = Vec::with_capacity(dimension * (dimension + 1) / 2);
    for column in 0..dimension {
        let rows = match triangle {
            Triangle::Lower => column..dimension,
            Triangle::Upper => 0..column + 1,
        };
        for row in rows {
            packed.push(matrix[(row, column)]);
        }
    }
    packed
}

fn validate_symmetric<T: ValidationScalar>(
    matrix: &DMatrix<T>,
    tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
) -> Result<usize, PackedMatrixError>
where
    <T as crate::LapackScalar>::Real: Float,
{
    let dimension = square_dimension(matrix)?;
    validate_tolerance(tolerance)?;
    for column in 0..dimension {
        for row in column + 1..dimension {
            if !approximately_equal(matrix[(row, column)], matrix[(column, row)], tolerance) {
                return Err(PackedMatrixError::NotSymmetric { row, column });
            }
        }
    }
    Ok(dimension)
}

fn validate_hermitian<T: ValidationScalar>(
    matrix: &DMatrix<T>,
    tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
) -> Result<usize, PackedMatrixError>
where
    <T as crate::LapackScalar>::Real: Float,
{
    let dimension = square_dimension(matrix)?;
    validate_tolerance(tolerance)?;
    for index in 0..dimension {
        if !approximately_real(matrix[(index, index)], tolerance) {
            return Err(PackedMatrixError::NonRealHermitianDiagonal { index });
        }
    }
    for column in 0..dimension {
        for row in column + 1..dimension {
            if !approximately_equal(
                matrix[(row, column)],
                ComplexField::conjugate(matrix[(column, row)]),
                tolerance,
            ) {
                return Err(PackedMatrixError::NotHermitian { row, column });
            }
        }
    }
    Ok(dimension)
}

#[derive(Clone, Copy)]
enum Reconstruction {
    Symmetric,
    Hermitian,
}

fn expand_lower_packed<T: StructuredScalar>(
    dimension: usize,
    packed: &[T],
    reconstruction: Reconstruction,
) -> DMatrix<T> {
    let mut data = vec![T::zero(); dimension * dimension];
    let mut packed_index = 0;
    for column in 0..dimension {
        for row in column..dimension {
            let value = packed[packed_index];
            packed_index += 1;
            if row == column {
                data[column * dimension + row] = match reconstruction {
                    Reconstruction::Symmetric => value,
                    Reconstruction::Hermitian => value.hermitian_diagonal(),
                };
            } else {
                data[column * dimension + row] = value;
                data[row * dimension + column] = match reconstruction {
                    Reconstruction::Symmetric => value,
                    Reconstruction::Hermitian => value.conjugate(),
                };
            }
        }
    }
    DMatrix::from_vec(dimension, dimension, data)
}

fn extract_lower<T: StructuredScalar>(
    matrix: &DMatrix<T>,
    canonicalize_diagonal: bool,
) -> Result<Vec<T>, PackedMatrixError> {
    let (rows, columns) = matrix.shape();
    if rows != columns {
        return Err(PackedMatrixError::NonSquareMatrix { rows, columns });
    }

    let mut packed = Vec::with_capacity(rows * (rows + 1) / 2);
    for column in 0..columns {
        for row in column..rows {
            let value = matrix[(row, column)];
            packed.push(if canonicalize_diagonal && row == column {
                value.hermitian_diagonal()
            } else {
                value
            });
        }
    }
    Ok(packed)
}

#[allow(private_bounds)]
impl<T, S> PackedSymmetric<T, S>
where
    T: StructuredScalar,
    S: PackedStorage<T>,
{
    /// Expands lower packed storage into a complete logical symmetric matrix.
    ///
    /// Both triangles are allocated in the result. Complex values are mirrored
    /// without conjugation, so complex symmetric matrices remain distinct from
    /// Hermitian matrices. The method borrows the source, copies its scalar values,
    /// allocates `n * n` column-major output, performs no validation, and calls no
    /// native backend. Supported scalars are `f32`, `f64`, `Complex32`, and `Complex64`.
    pub fn to_dmatrix(&self) -> DMatrix<T> {
        expand_lower_packed(self.dimension(), self.as_slice(), Reconstruction::Symmetric)
    }
}

#[allow(private_bounds)]
impl<T> PackedSymmetric<T>
where
    T: StructuredScalar,
{
    /// Extracts the lower triangle of a square matrix without validating symmetry.
    ///
    /// The upper triangle is intentionally ignored and the result owns newly
    /// allocated traditional lower column-packed storage. The input is borrowed and
    /// scalars are copied; only square shape is validated. A rectangular matrix
    /// returns [`PackedMatrixError::NonSquareMatrix`]. This pure-Rust path supports
    /// the four matrixpacked scalar types and has no backend requirement.
    pub fn from_lower_triangle(matrix: &DMatrix<T>) -> Result<Self, PackedMatrixError> {
        Self::from_vec(matrix.nrows(), extract_lower(matrix, false)?)
    }
}

#[allow(private_bounds)]
impl<T, S> PackedHermitian<T, S>
where
    T: StructuredScalar,
    S: PackedStorage<T>,
{
    /// Expands lower packed storage into a complete logical Hermitian matrix.
    ///
    /// Upper entries are conjugated and diagonal imaginary components are
    /// discarded according to LAPACK's real-diagonal convention. The result
    /// borrows and copies the source into allocated `n * n` column-major storage,
    /// performs no validation, and calls no native backend. Supported scalars are
    /// `f32`, `f64`, `Complex32`, and `Complex64`.
    pub fn to_dmatrix(&self) -> DMatrix<T> {
        expand_lower_packed(self.dimension(), self.as_slice(), Reconstruction::Hermitian)
    }
}

#[allow(private_bounds)]
impl<T> PackedHermitian<T>
where
    T: StructuredScalar,
{
    /// Extracts the lower triangle of a square matrix without validating Hermitian structure.
    ///
    /// The upper triangle is intentionally ignored. Diagonal imaginary components
    /// are discarded to retain LAPACK's real-diagonal convention. The borrowed input
    /// is copied into newly allocated traditional lower column-packed storage. Only
    /// square shape is validated; this pure-Rust path supports the four matrixpacked
    /// scalar types and has no backend requirement.
    pub fn from_lower_triangle(matrix: &DMatrix<T>) -> Result<Self, PackedMatrixError> {
        Self::from_vec(matrix.nrows(), extract_lower(matrix, true)?)
    }
}

#[allow(private_bounds)]
impl<T, S> PackedSPD<T, S>
where
    T: StructuredScalar,
    S: PackedStorage<T>,
{
    /// Expands lower packed SPD/HPD storage into a complete logical matrix.
    ///
    /// Real values are reconstructed symmetrically; complex values use Hermitian
    /// conjugation and a real diagonal. The conversion allocates `n * n` entries
    /// and cannot provide a zero-copy view. It borrows and copies the source, does
    /// not validate the SPD/HPD claim, produces column-major output, and calls no
    /// native backend. Supported scalars are the four matrixpacked scalar types.
    pub fn to_dmatrix(&self) -> DMatrix<T> {
        expand_lower_packed(self.dimension(), self.as_slice(), Reconstruction::Hermitian)
    }
}

#[allow(private_bounds)]
impl<T> PackedSPD<T>
where
    T: StructuredScalar,
{
    /// Extracts lower storage without proving symmetric/Hermitian or positive-definite structure.
    ///
    /// The upper triangle is intentionally ignored. Complex diagonal imaginary
    /// components are discarded according to LAPACK's HPD convention. Numerical
    /// invalidity is possible but does not affect memory safety. The borrowed input
    /// is copied into newly allocated lower column-packed storage. Only square shape
    /// is checked; this supports the four matrixpacked scalars and calls no backend.
    pub fn from_lower_triangle_unchecked_structure(
        matrix: &DMatrix<T>,
    ) -> Result<Self, PackedMatrixError> {
        Self::from_vec(matrix.nrows(), extract_lower(matrix, true)?)
    }
}

#[allow(private_bounds)]
impl<T> PackedLower<T>
where
    T: ValidationScalar,
    <T as crate::LapackScalar>::Real: Float,
{
    /// Validates and packs a square lower-triangular nalgebra matrix.
    ///
    /// Unlike [`Self::from_lower_triangle`], this rejects an upper-triangle
    /// entry unless it is approximately zero under `tolerance`. The borrowed input
    /// is copied into newly allocated lower column-packed storage. Errors report a
    /// non-square matrix, invalid tolerance, structural mismatch, or packed dimension
    /// overflow. This pure-Rust path supports all four matrixpacked scalar types and
    /// does not call LAPACK.
    pub fn try_from_dmatrix(
        matrix: &DMatrix<T>,
        tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
    ) -> Result<Self, PackedMatrixError> {
        let dimension = validate_triangle(matrix, Triangle::Lower, tolerance)?;
        Self::from_vec(dimension, pack_triangle(matrix, Triangle::Lower))
    }
}

#[allow(private_bounds)]
impl<T> PackedUpper<T>
where
    T: ValidationScalar,
    <T as crate::LapackScalar>::Real: Float,
{
    /// Validates and packs a square upper-triangular nalgebra matrix.
    ///
    /// Unlike [`Self::from_upper_triangle`], this rejects a lower-triangle
    /// entry unless it is approximately zero under `tolerance`. The borrowed input
    /// is copied into newly allocated upper column-packed storage. Errors report a
    /// non-square matrix, invalid tolerance, structural mismatch, or packed dimension
    /// overflow. This pure-Rust path supports all four matrixpacked scalar types and
    /// does not call LAPACK.
    pub fn try_from_dmatrix(
        matrix: &DMatrix<T>,
        tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
    ) -> Result<Self, PackedMatrixError> {
        let dimension = validate_triangle(matrix, Triangle::Upper, tolerance)?;
        Self::from_vec(dimension, pack_triangle(matrix, Triangle::Upper))
    }
}

#[allow(private_bounds)]
impl<T> PackedSymmetric<T>
where
    T: ValidationScalar,
    <T as crate::LapackScalar>::Real: Float,
{
    /// Validates symmetry and stores the lower triangle exactly as supplied.
    ///
    /// Complex symmetry compares entries directly without conjugation. The
    /// upper triangle is validation evidence only and is never averaged into
    /// the stored values. The borrowed input is copied into newly allocated lower
    /// column-packed storage. Errors report non-square input, invalid tolerance, or
    /// a symmetry mismatch. This pure-Rust path supports all four matrixpacked scalar
    /// types and has no backend requirement.
    pub fn try_from_dmatrix(
        matrix: &DMatrix<T>,
        tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
    ) -> Result<Self, PackedMatrixError> {
        let dimension = validate_symmetric(matrix, tolerance)?;
        Self::from_vec(dimension, extract_lower(matrix, false)?)
    }
}

#[allow(private_bounds)]
impl<T> PackedHermitian<T>
where
    T: ValidationScalar,
    <T as crate::LapackScalar>::Real: Float,
{
    /// Validates Hermitian structure and stores the lower triangle.
    ///
    /// Accepted small imaginary diagonal components are normalized to zero.
    /// Off-diagonal upper values are used only for validation; lower values are
    /// preserved exactly. The borrowed input is copied into newly allocated lower
    /// column-packed storage. Errors report non-square input, invalid tolerance,
    /// non-real diagonal, or a Hermitian mismatch. This pure-Rust path supports all
    /// four matrixpacked scalar types and has no backend requirement.
    pub fn try_from_dmatrix(
        matrix: &DMatrix<T>,
        tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
    ) -> Result<Self, PackedMatrixError> {
        let dimension = validate_hermitian(matrix, tolerance)?;
        Self::from_vec(dimension, extract_lower(matrix, true)?)
    }
}

#[allow(private_bounds)]
impl<T> PackedSPD<T>
where
    T: ValidationScalar,
    <T as crate::LapackScalar>::Real: Float,
{
    /// Validates symmetric/Hermitian structure without proving positive definiteness.
    ///
    /// Real matrices are checked for symmetry; complex matrices are checked for
    /// Hermitian structure. The lower triangle is retained exactly except that
    /// accepted complex diagonal noise is normalized to zero. It borrows the input
    /// and allocates lower column-packed storage, but deliberately does not prove
    /// positive definiteness. Errors cover shape, tolerance, and structural failures.
    /// All four matrixpacked scalars are supported; no native backend is called.
    pub fn try_from_structured_dmatrix(
        matrix: &DMatrix<T>,
        tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
    ) -> Result<Self, PackedMatrixError> {
        let dimension = if T::IS_COMPLEX {
            validate_hermitian(matrix, tolerance)?
        } else {
            validate_symmetric(matrix, tolerance)?
        };
        Self::from_vec(dimension, extract_lower(matrix, true)?)
    }

    /// Validates structure and positive definiteness before packing the lower triangle.
    ///
    /// After tolerance-aware structural validation, this reconstructs a canonical
    /// full matrix from the selected lower triangle and runs nalgebra Cholesky.
    /// The operation uses `O(n^3)` work and `O(n^2)` temporary storage. It does
    /// not call matrixpacked's LAPACK factorization. It borrows the input and allocates
    /// both lower column-packed output and temporary column-major full storage. Errors
    /// cover shape, tolerance, structural failures, and
    /// [`PackedMatrixError::NotPositiveDefinite`]. All four matrixpacked scalars are
    /// supported and no native backend is required.
    pub fn try_from_dmatrix(
        matrix: &DMatrix<T>,
        tolerance: ConversionTolerance<<T as crate::LapackScalar>::Real>,
    ) -> Result<Self, PackedMatrixError> {
        let packed = Self::try_from_structured_dmatrix(matrix, tolerance)?;
        let Some(cholesky) = Cholesky::new(packed.to_dmatrix()) else {
            return Err(PackedMatrixError::NotPositiveDefinite);
        };
        let factor = cholesky.unpack_dirty();
        for index in 0..factor.nrows() {
            let diagonal = factor[(index, index)];
            if !diagonal.real().is_finite()
                || !diagonal.imaginary().is_zero()
                || diagonal.real() <= <T as crate::LapackScalar>::Real::zero()
            {
                return Err(PackedMatrixError::NotPositiveDefinite);
            }
        }
        Ok(packed)
    }
}
