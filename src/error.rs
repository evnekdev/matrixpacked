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

    StructuralZero { row: usize, col: usize },
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidIncrement { increment: i32 },
    DimensionMismatch { left: usize, right: usize },
    LapackIllegalArgument { argument: i32 },
    FactorizationFailure { index: usize, message: &'static str },
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

            Self::InvalidVectorLength { expected, actual } => write!(f, "invalid vector length: expected {expected}, got {actual}"),
            Self::InvalidIncrement { increment } => write!(f, "BLAS vector increment must be nonzero, got {increment}"),
            Self::DimensionMismatch { left, right } => write!(f, "matrix dimensions differ: {left} and {right}"),
            Self::LapackIllegalArgument { argument } => write!(f, "LAPACK reported an invalid argument at position {argument}"),
            Self::FactorizationFailure { index, message } => write!(f, "{message} (leading index {index})"),
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
