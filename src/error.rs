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
                write!(
                    f,
                    "matrix index ({row}, {col}) is outside a {n}x{n} matrix"
                )
            }

            Self::StructuralZero { row, col } => {
                write!(
                    f,
                    "element ({row}, {col}) is outside the stored lower triangle"
                )
            }
        }
    }
}

impl std::error::Error for PackedMatrixError {}