// packedmatrix::error.rs

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedMatrixError {
    DimensionOverflow {
        n: usize,
    },

    InvalidLength {
        n: usize,
        expected: usize,
        actual: usize,
    },

    IndexOutOfBounds {
        row: usize,
        col: usize,
        n: usize,
    },

    StructuralZero {
        row: usize,
        col: usize,
    },
    InvalidVectorLength {
        expected: usize,
        actual: usize,
    },
    InvalidIncrement {
        increment: i32,
    },
    DimensionMismatch {
        left: usize,
        right: usize,
    },
    /// A conversion requiring a square matrix received a rectangular matrix.
    NonSquareMatrix {
        rows: usize,
        columns: usize,
    },
    /// A conversion tolerance was negative or non-finite.
    InvalidTolerance {
        component: &'static str,
        reason: &'static str,
    },
    /// A value in the opposite triangle was not approximately zero.
    NotTriangular {
        triangle: &'static str,
        row: usize,
        column: usize,
    },
    /// Opposite entries did not satisfy the symmetric relation.
    NotSymmetric {
        row: usize,
        column: usize,
    },
    /// Opposite entries did not satisfy the Hermitian relation.
    NotHermitian {
        row: usize,
        column: usize,
    },
    /// A Hermitian diagonal entry had excessive imaginary magnitude.
    NonRealHermitianDiagonal {
        index: usize,
    },
    /// Nalgebra Cholesky rejected a structurally valid SPD/HPD matrix.
    NotPositiveDefinite,
    TriangleMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    LapackIllegalArgument {
        argument: i32,
    },
    EigenvalueConvergenceFailure {
        unconverged: usize,
    },
    InvalidWorkspaceRecommendation {
        workspace: &'static str,
    },
    InvalidEigenRange {
        reason: &'static str,
    },
    EigenvectorConvergenceFailure {
        failed: Vec<usize>,
    },
    PositiveDefinitenessFailure {
        index: usize,
    },
    NonPositiveDiagonal {
        index: usize,
    },
    InvalidLeadingDimension {
        minimum: usize,
        actual: usize,
    },
    FactorizationFailure {
        index: usize,
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
