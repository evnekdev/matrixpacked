// packedmatrix::error.rs

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Errors produced by packed-storage validation and BLAS/LAPACK operations.
///
/// Dimension and layout errors can normally be corrected before calling the
/// numerical routine. Numerical failures describe the LAPACK condition that
/// prevented a factorization, solve, or eigensolver from completing.
pub enum PackedMatrixError {
    /// Computing the packed length `n * (n + 1) / 2` overflowed `usize`.
    DimensionOverflow {
        /// Requested square-matrix dimension.
        n: usize,
    },

    /// A packed buffer did not contain exactly `n * (n + 1) / 2` elements.
    InvalidLength {
        /// Declared square-matrix dimension.
        n: usize,
        /// Required number of packed elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },

    /// A zero-based logical matrix coordinate was outside the square matrix.
    IndexOutOfBounds {
        /// Requested row.
        row: usize,
        /// Requested column.
        col: usize,
        /// Matrix dimension.
        n: usize,
    },

    /// Mutable stored-only access targeted an implicit triangular zero.
    StructuralZero {
        /// Requested row.
        row: usize,
        /// Requested column.
        col: usize,
    },
    /// A vector or column-major right-hand-side buffer had the wrong length.
    InvalidVectorLength {
        /// Required number of elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },
    /// A BLAS vector stride was zero; use any nonzero positive or negative stride.
    InvalidIncrement {
        /// Invalid stride supplied by the caller.
        increment: i32,
    },
    /// Two packed operands had different square dimensions.
    DimensionMismatch {
        /// Dimension of the left operand.
        left: usize,
        /// Dimension of the right operand.
        right: usize,
    },
    /// A conversion requiring a square matrix received a rectangular matrix.
    NonSquareMatrix {
        /// Number of source rows.
        rows: usize,
        /// Number of source columns.
        columns: usize,
    },
    /// A conversion tolerance was negative or non-finite.
    InvalidTolerance {
        /// Tolerance component, such as `absolute` or `relative`.
        component: &'static str,
        /// Why the component is invalid.
        reason: &'static str,
    },
    /// A value in the opposite triangle was not approximately zero.
    NotTriangular {
        /// Expected stored triangle.
        triangle: &'static str,
        /// Row of the offending opposite-triangle entry.
        row: usize,
        /// Column of the offending opposite-triangle entry.
        column: usize,
    },
    /// Opposite entries did not satisfy the symmetric relation.
    NotSymmetric {
        /// Row of the first mismatched entry.
        row: usize,
        /// Column of the first mismatched entry.
        column: usize,
    },
    /// Opposite entries did not satisfy the Hermitian relation.
    NotHermitian {
        /// Row of the first mismatched entry.
        row: usize,
        /// Column of the first mismatched entry.
        column: usize,
    },
    /// A Hermitian diagonal entry had excessive imaginary magnitude.
    NonRealHermitianDiagonal {
        /// Zero-based diagonal position with excessive imaginary magnitude.
        index: usize,
    },
    /// Nalgebra Cholesky rejected a structurally valid SPD/HPD matrix.
    NotPositiveDefinite,
    /// A conversion was requested for a different stored triangle.
    TriangleMismatch {
        /// Triangle required by the destination.
        expected: &'static str,
        /// Triangle carried by the source value.
        actual: &'static str,
    },
    /// LAPACK reported an illegal argument, indicating invalid routine inputs.
    LapackIllegalArgument {
        /// One-based LAPACK argument position (reported as a positive value).
        argument: i32,
    },
    /// A standard eigensolver failed to converge completely.
    EigenvalueConvergenceFailure {
        /// Number of off-diagonal elements that did not converge.
        unconverged: usize,
    },
    /// A LAPACK workspace query returned a non-finite or unusable size.
    InvalidWorkspaceRecommendation {
        /// Workspace array whose recommendation was invalid.
        workspace: &'static str,
    },
    /// An [`crate::EigenRange`] had invalid bounds or indices.
    InvalidEigenRange {
        /// Human-readable range validation failure.
        reason: &'static str,
    },
    /// A selected eigensolver could not converge one or more eigenvectors.
    EigenvectorConvergenceFailure {
        /// Zero-based indices of failed eigenvectors.
        failed: Vec<usize>,
    },
    /// The positive-definite `B` factor in a generalized problem failed.
    PositiveDefinitenessFailure {
        /// One-based leading principal minor that was not positive definite.
        index: usize,
    },
    /// Packed positive-definite equilibration encountered a non-positive diagonal.
    NonPositiveDiagonal {
        /// Zero-based offending diagonal position.
        index: usize,
    },
    /// A full column-major matrix used an insufficient leading dimension.
    InvalidLeadingDimension {
        /// Smallest accepted leading dimension.
        minimum: usize,
        /// Supplied leading dimension.
        actual: usize,
    },
    /// A packed factorization found a singular or invalid leading block.
    FactorizationFailure {
        /// One-based leading index reported by LAPACK.
        index: usize,
        /// Factorization-specific explanation.
        message: &'static str,
    },
}

impl fmt::Display for PackedMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow { n } => {
                write!(f, "packed matrix size overflows for dimension {n}")
            }

            Self::InvalidLength {
                n,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "invalid packed data length for {n}x{n} matrix: \
                     expected {expected}, got {actual}"
                )
            }

            Self::IndexOutOfBounds { row, col, n } => {
                write!(f, "matrix index ({row}, {col}) is outside a {n}x{n} matrix")
            }

            Self::InvalidVectorLength { expected, actual } => write!(
                f,
                "invalid vector length: expected {expected}, got {actual}"
            ),
            Self::InvalidIncrement { increment } => {
                write!(f, "BLAS vector increment must be nonzero, got {increment}")
            }
            Self::DimensionMismatch { left, right } => {
                write!(f, "matrix dimensions differ: {left} and {right}")
            }
            Self::NonSquareMatrix { rows, columns } => {
                write!(f, "matrix must be square, got {rows}x{columns}")
            }
            Self::InvalidTolerance { component, reason } => {
                write!(f, "invalid {component} conversion tolerance: {reason}")
            }
            Self::NotTriangular {
                triangle,
                row,
                column,
            } => write!(
                f,
                "matrix is not {triangle} triangular: entry ({row}, {column}) is not approximately zero"
            ),
            Self::NotSymmetric { row, column } => write!(
                f,
                "matrix is not symmetric: entries ({row}, {column}) and ({column}, {row}) differ"
            ),
            Self::NotHermitian { row, column } => write!(
                f,
                "matrix is not Hermitian: entries ({row}, {column}) and ({column}, {row}) are not conjugates"
            ),
            Self::NonRealHermitianDiagonal { index } => write!(
                f,
                "matrix is not Hermitian: diagonal entry ({index}, {index}) is not approximately real"
            ),
            Self::NotPositiveDefinite => {
                write!(f, "matrix is not positive definite")
            }
            Self::TriangleMismatch { expected, actual } => {
                write!(f, "triangle mismatch: expected {expected}, got {actual}")
            }
            Self::LapackIllegalArgument { argument } => write!(
                f,
                "LAPACK reported an invalid argument at position {argument}"
            ),
            Self::EigenvalueConvergenceFailure { unconverged } => write!(
                f,
                "LAPACK failed to converge for {unconverged} off-diagonal elements"
            ),
            Self::InvalidWorkspaceRecommendation { workspace } => write!(
                f,
                "LAPACK returned an invalid {workspace} workspace recommendation"
            ),
            Self::InvalidEigenRange { reason } => {
                write!(f, "invalid eigenvalue selection range: {reason}")
            }
            Self::EigenvectorConvergenceFailure { failed } => write!(
                f,
                "LAPACK failed to converge for eigenvectors at indices {failed:?}"
            ),
            Self::PositiveDefinitenessFailure { index } => write!(
                f,
                "the B matrix is not positive definite (leading principal minor {index})"
            ),
            Self::NonPositiveDiagonal { index } => {
                write!(f, "matrix diagonal element {index} is non-positive")
            }
            Self::InvalidLeadingDimension { minimum, actual } => write!(
                f,
                "invalid leading dimension: expected at least {minimum}, got {actual}"
            ),
            Self::FactorizationFailure { index, message } => {
                write!(f, "{message} (leading index {index})")
            }
            Self::StructuralZero { row, col } => {
                write!(
                    f,
                    "element ({row}, {col}) is not physically stored by this packed matrix"
                )
            }
        }
    }
}

impl std::error::Error for PackedMatrixError {}
