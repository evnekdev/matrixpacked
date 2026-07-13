// lib.rs

//! Triangularly packed matrix representations with direct BLAS/LAPACK packed-format operations.
//!
//! Enable `openblas-static` to bundle an OpenBLAS provider, use `intel-mkl-static`
//! for a Windows-compatible bundled provider, or link another compatible BLAS/LAPACK
//! implementation in the final application.

mod arithmetic;
mod backend;
mod conversions;
mod diagnostics;
mod eigen;
mod equilibration;
pub mod error;
mod expert_solve;
pub mod factorization;
mod formatting;
mod generalized_reduction;
mod norms;
mod rank_updates;
pub mod scalar;
mod simple_solve;
pub mod storage;
pub mod triangular;
mod tridiagonal;

pub use conversions::{
    FullTriangular, RectangularFullPacked, RectangularFullPackedView, RectangularFullPackedViewMut,
    RfpTranspose, Triangle,
};
pub use diagnostics::{Inertia, SignedLogDet};
pub use equilibration::Equilibration;
pub use expert_solve::{EquilibrationMode, ExpertSolveOptions, ExpertSolveResult};
pub use tridiagonal::{
    ApplySide, HermitianPackedTridiagonal, OrthogonalOperation, SymmetricPackedTridiagonal,
    UnitaryOperation,
};

#[cfg(feature = "openblas-static")]
use openblas_src as _;

#[cfg(feature = "intel-mkl-static")]
use intel_mkl_src as _;

pub mod hermitian;
pub mod lower;
pub mod spd;
pub mod symmetric;
pub mod upper;

pub use eigen::{
    EigenDecomposition, EigenRange, Eigenvectors, GeneralizedEigenproblem,
    SelectedEigenDecomposition,
};
pub use error::PackedMatrixError;
pub use hermitian::{PackedHermitian, PackedHermitianView, PackedHermitianViewMut};
pub use lower::{PackedLower, PackedLowerView, PackedLowerViewMut};
pub use scalar::LapackScalar;
pub use spd::{PackedSPD, PackedSPDView, PackedSPDViewMut};
pub use symmetric::{PackedSymmetric, PackedSymmetricView, PackedSymmetricViewMut};
pub use triangular::{ConditionNorm, Diagonal, MatrixNorm, RefinementReport, Transpose};
pub use upper::{PackedUpper, PackedUpperView, PackedUpperViewMut};

pub use factorization::{
    PackedCholesky, PackedCholeskyViewMut, PackedHermitianFactor, PackedHermitianFactorViewMut,
    PackedSymmetricFactor, PackedSymmetricFactorViewMut,
};
