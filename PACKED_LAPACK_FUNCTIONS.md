# Packed BLAS/LAPACK function status

This document tracks BLAS and LAPACK routines that operate directly on traditional packed triangular, symmetric, positive-definite, and Hermitian storage, together with their implementation status in `matrixpacked`.

Factorization creates a reusable packed factor, while solve applies that factor
to right-hand sides. Iterative refinement is a separate operation requiring
both the unchanged original matrix and its factorization: it modifies an
existing solution in place and returns one forward error estimate (relative
solution error bound) and one backward error estimate (relative residual) per
column-major right-hand side.

It is intended as a roadmap and reference. Status should be updated when public APIs and examples are added.

Inverse APIs use a consistent ownership vocabulary: `inverse_in_place()`
overwrites writable packed storage, `into_inverse()` consumes an owned matrix or
factorization and returns the structured packed inverse, and `inverse()` borrows
and allocates only packed storage. Factorization `inverse_in_place()` methods are
retained for mutable-view workflows; after success their storage contains the
inverse and no longer represents the original factorization.

## Status legend

| Status | Meaning |
|---|---|
| **Implemented** | Backend dispatch and a public API exist. |
| **Backend only** | Internal binding exists, but no complete public API is exposed. |
| **Missing** | Exposed by the selected Rust `lapack` or `blas` crate but not implemented in this crate. |
| **Optional** | Useful interoperability or low-level functionality, but not central to the packed matrix abstraction. |
| **Not applicable** | The operation does not match the matrix structure or has no traditional packed-storage routine. |

`matrixpacked` intentionally uses only the Rust `lapack` and `blas` crates for native routine declarations. LAPACK routines not exposed by those crates are not bound manually.

The routine prefix `x` represents the scalar family:

| Prefix | Rust scalar |
|---|---|
| `S` | `f32` |
| `D` | `f64` |
| `C` | `Complex<f32>` |
| `Z` | `Complex<f64>` |

---

## Packed triangular matrices

Applies to `PackedLower<T>` and `PackedUpper<T>`.

| Function family | Short description | Implementation status |
|---|---|---|
| `xTPMV` | In-place packed triangular matrix-vector multiplication, with transpose/conjugate-transpose and unit-diagonal modes. | **Implemented** |
| `xTPSV` | Single-vector packed triangular solve through BLAS. | **Implemented** |
| `xTPTRS` | Solve one or more right-hand sides with a packed triangular matrix. | **Implemented** |
| `xTPTRI` | Compute the inverse of a packed triangular matrix in place. | **Implemented** |
| `xLATPS` | Overflow-safe packed triangular solve with scaling. | **Unsupported by the selected Rust `lapack` crate; no custom FFI bindings are maintained.** |
| `xTPCON` | Estimate the reciprocal condition number without forming the inverse. | **Implemented** |
| `xTPRFS` | Iterative refinement and forward/backward error estimates. | **Implemented** |
| `xLANTP` | Compute max, one, infinity, or Frobenius norm. | **Implemented** |
| `xTPTTR` | Convert traditional packed triangular storage to full triangular storage. | **Missing / optional** |
| `xTRTTP` | Convert full triangular storage to traditional packed storage. | **Missing / optional** |
| `xTPTTF` | Convert traditional packed triangular storage to rectangular full packed storage. | **Missing / optional** |
| `xTFTTP` | Convert rectangular full packed storage to traditional packed storage. | **Missing / optional** |

### Naming warning

LAPACK routines such as `xTPQRT`, `xTPMQRT`, `xTPLQT`, and `xTPMLQT` are triangular-pentagonal QR/LQ routines. Their `TP` substring does **not** mean traditional packed triangular storage, so they do not belong to `PackedLower` or `PackedUpper`.

---

## Real symmetric packed matrices

Applies primarily to `PackedSymmetric<f32>` and `PackedSymmetric<f64>`.

### BLAS operations

| Function family | Short description | Implementation status |
|---|---|---|
| `xSPMV` | Symmetric packed matrix-vector multiplication. | **Implemented** |
| `xSPR` | Symmetric packed rank-1 update: `A := A + alpha*x*x^T`. | **Implemented** (`f32`, `f64`) |
| `xSPR2` | Symmetric packed rank-2 update. | **Implemented** (`f32`, `f64`) |

### Linear systems, conditioning, and inverse

| Function family | Short description | Implementation status |
|---|---|---|
| `xSPTRF` | Bunch-Kaufman factorization of a symmetric indefinite packed matrix. | **Implemented** |
| `xSPTRS` | Solve from a packed Bunch-Kaufman factorization. | **Implemented** |
| `xSPTRI` | Compute the inverse from a packed Bunch-Kaufman factorization. | **Implemented** |
| `xSPSV` | One-shot packed factor-and-solve driver. | **Missing** |
| `xSPSVX` | Expert factor-and-solve driver with condition estimate, refinement, and error bounds. | **Missing** |
| `xSPCON` | Estimate reciprocal condition number from an `xSPTRF` factorization. | **Implemented** |
| `xSPRFS` | Iterative refinement and forward/backward error estimates. | **Implemented** (`s`, `d`, `c`, `z`; exposed by `lapack` 0.20) |
| `xLANSP` | Compute the norm of a real symmetric packed matrix. | **Implemented** (`f32`, `f64`; complex-symmetric backend dispatch also exists) |

### Eigenvalues and eigenvectors

| Function family | Short description | Implementation status |
|---|---|---|
| `xSPEV` | Compute all eigenvalues and optionally eigenvectors using the classical driver. | **Implemented** (`f32`, `f64`) |
| `xSPEVD` | Compute all eigenvalues and optionally eigenvectors using divide-and-conquer. | **Implemented** (`f32`, `f64`) |
| `xSPEVX` | Compute selected eigenvalues/eigenvectors by index or value range. | **Implemented** (`f32`, `f64`) |
| `xSPTRD` | Reduce a real symmetric packed matrix to real symmetric tridiagonal form. | **Missing** |
| `xOPGTR` | Generate the orthogonal transformation from `xSPTRD` reflectors. | **Missing** |
| `xOPMTR` | Apply the `xSPTRD` orthogonal transformation to another matrix. | **Missing** |

### Generalized symmetric-definite eigenproblems

These routines solve problems involving a symmetric packed `A` and SPD packed `B`, for example `A*x = lambda*B*x`.

| Function family | Short description | Implementation status |
|---|---|---|
| `xSPGV` | Generalized symmetric-definite eigenproblem. | **Implemented** (`f32`, `f64`) |
| `xSPGVD` | Divide-and-conquer generalized symmetric-definite eigenproblem. | **Implemented** (`f32`, `f64`) |
| `xSPGVX` | Selected generalized eigenvalues/eigenvectors. | **Implemented** (`f32`, `f64`) |
| `xSPGST` | Reduce a generalized packed problem to a standard symmetric eigenproblem. | **Missing** |

---

## Symmetric/Hermitian positive-definite packed matrices

Applies to `PackedSPD<T>`. For real scalars, the matrix is symmetric positive definite. For complex scalars, it is Hermitian positive definite.

### Core factorization and solving

| Function family | Short description | Implementation status |
|---|---|---|
| `xPPTRF` | Packed Cholesky factorization. | **Implemented** |
| `xPPTRS` | Solve from a packed Cholesky factorization. | **Implemented** |
| `xPPTRI` | Compute the inverse from a packed Cholesky factor. | **Implemented** |
| `xPPSV` | One-shot packed Cholesky factor-and-solve driver. | **Missing** |
| `xPPSVX` | Expert packed SPD/HPD driver with equilibration, condition estimate, refinement, and error bounds. | **Missing** |
| `xPPCON` | Estimate reciprocal condition number from the packed Cholesky factor. | **Implemented** |
| `xPPEQU` | Compute row/column scaling factors for equilibration. | **Implemented** (`s`, `d`, `c`, `z`) |
| `xPPRFS` | Iterative refinement and forward/backward error estimates. | **Implemented** (`s`, `d`, `c`, `z`) |

### Multiplication, updates, and norms

| Function family | Short description | Implementation status |
|---|---|---|
| `xSPMV` | Real SPD packed matrix-vector multiplication. | **Implemented** |
| `xHPMV` | Complex HPD packed matrix-vector multiplication. | **Implemented** |
| `xSPR` / `xHPR` | Symmetric/Hermitian packed rank-1 update. | **Implemented** for every applicable scalar |
| `xSPR2` / `xHPR2` | Symmetric/Hermitian packed rank-2 update. | **Implemented** for every applicable scalar |
| `xLANSP` / `xLANHP` | Packed symmetric/Hermitian matrix norm. | **Implemented** for every applicable scalar (`f32`, `f64` / `Complex32`, `Complex64`) |

### Eigenvalues

There is no distinct `PPEV` family. An SPD/HPD matrix uses the symmetric or Hermitian packed eigenvalue drivers:

| Matrix scalar | Eigenvalue families | Implementation status |
|---|---|---|
| `f32`, `f64` | `xSPEV`, `xSPEVD`, `xSPEVX` | **Implemented** |
| `Complex<f32>`, `Complex<f64>` | `xHPEV`, `xHPEVD`, `xHPEVX` | **Implemented** |

A rank update must preserve positive definiteness. A public SPD rank-update API should therefore either validate/update under a restricted coefficient, or return a less restrictive symmetric/Hermitian packed type.

---

## Complex Hermitian packed matrices

Applies to `PackedHermitian<Complex<f32>>` and `PackedHermitian<Complex<f64>>`.

### BLAS operations

| Function family | Short description | Implementation status |
|---|---|---|
| `xHPMV` | Hermitian packed matrix-vector multiplication. | **Implemented** |
| `xHPR` | Hermitian packed rank-1 update: `A := A + alpha*x*x^H`, with real `alpha`. | **Implemented** (`Complex32`, `Complex64`) |
| `xHPR2` | Hermitian packed rank-2 update. | **Implemented** (`Complex32`, `Complex64`) |

### Linear systems, conditioning, and inverse

| Function family | Short description | Implementation status |
|---|---|---|
| `xHPTRF` | Bunch-Kaufman factorization of a Hermitian indefinite packed matrix. | **Implemented** |
| `xHPTRS` | Solve from a packed Hermitian factorization. | **Implemented** |
| `xHPTRI` | Compute the inverse from a packed Hermitian factorization. | **Implemented** |
| `xHPSV` | One-shot packed factor-and-solve driver. | **Missing** |
| `xHPSVX` | Expert packed Hermitian driver with condition estimate and refinement. | **Missing** |
| `xHPCON` | Estimate reciprocal condition number from the packed factorization. | **Implemented** |
| `xHPRFS` | Iterative refinement and forward/backward error estimates. | **Implemented** (`c`, `z`) |
| `xLANHP` | Compute the norm of a Hermitian packed matrix. | **Implemented** (`Complex32`, `Complex64`) |

### Eigenvalues and eigenvectors

| Function family | Short description | Implementation status |
|---|---|---|
| `xHPEV` | Compute all Hermitian eigenvalues and optionally eigenvectors. | **Implemented** (`Complex32`, `Complex64`) |
| `xHPEVD` | Divide-and-conquer Hermitian eigensolver. | **Implemented** (`Complex32`, `Complex64`) |
| `xHPEVX` | Compute selected Hermitian eigenvalues/eigenvectors. | **Implemented** (`Complex32`, `Complex64`) |
| `xHPTRD` | Reduce a Hermitian packed matrix to real symmetric tridiagonal form. | **Missing** |
| `xUPGTR` | Generate the unitary transformation from `xHPTRD` reflectors. | **Missing** |
| `xUPMTR` | Apply the packed unitary transformation to another matrix. | **Missing** |

The eigenvalues returned by the Hermitian packed eigenvalue routines are real.

### Generalized Hermitian-definite eigenproblems

| Function family | Short description | Implementation status |
|---|---|---|
| `xHPGV` | Generalized Hermitian-definite eigenproblem. | **Implemented** (`Complex32`, `Complex64`) |
| `xHPGVD` | Divide-and-conquer generalized Hermitian-definite eigenproblem. | **Implemented** (`Complex32`, `Complex64`) |
| `xHPGVX` | Selected generalized eigenvalues/eigenvectors. | **Implemented** (`Complex32`, `Complex64`) |
| `xHPGST` | Reduce a generalized packed problem to standard Hermitian form. | **Missing** |

---

## Complex symmetric packed matrices

`PackedSymmetric<Complex<_>>` represents a complex symmetric matrix (`A^T = A`), not a Hermitian matrix (`A^H = A`).

| Function family | Short description | Implementation status |
|---|---|---|
| `xSPTRF` | Complex-symmetric packed factorization (`CSPTRF` / `ZSPTRF`). | **Implemented** |
| `xSPTRS` | Solve from a complex-symmetric packed factorization. | **Implemented** |
| `xSPTRI` | Inverse from a complex-symmetric packed factorization. | **Implemented** |
| `xSPCON` | Reciprocal condition estimate from a complex-symmetric packed factorization. | **Implemented** (`Complex32`, `Complex64`) |
| `xSPRFS` | Iterative refinement and forward/backward error estimates. | **Implemented** (`Complex32`, `Complex64`) |
| `xSPSV`, `xSPSVX` | Simple and expert factor-and-solve drivers. | **Missing but exposed by the selected Rust `lapack` crate** (`cspsv`/`zspsv`, `cspsvx`/`zspsvx`) |
| Packed Hermitian eigensolvers | Not valid for a merely complex-symmetric matrix. | **Not applicable** |

Complex symmetric matrices do not generally have real eigenvalues or unitary eigenvectors. A general complex eigensolver normally requires conversion to general dense storage.

---

## Operations not provided for traditional packed storage

| Operation | Notes |
|---|---|
| Packed matrix × packed matrix | Traditional packed BLAS has no Level-3 packed matrix-matrix routine. |
| Packed matrix × dense matrix | No dedicated traditional packed Level-3 routine. Column-by-column Level-2 calls are possible but are not equivalent to GEMM performance. |
| SVD directly from packed symmetric/Hermitian storage | No packed SVD driver. |
| General nonsymmetric eigenproblem in packed format | Traditional packed formats represent structured matrices only. |
| General QR/LU factorization | Not defined for these structured packed matrix types. |
| Determinant/log-determinant driver | Derive from Cholesky or LDLT/Bunch-Kaufman factors. |
| Exact rank driver | Usually estimate from eigenvalues or a factorization with a tolerance. |

Traditional packed storage minimizes memory but cannot use most Level-3 BLAS kernels. Rectangular Full Packed storage is a separate LAPACK format designed to retain `n*(n+1)/2` elements while enabling more Level-3 operations.

---

## Recommended implementation order

### Priority 1: simple and expert solve drivers

Add `xPPSV`, `xSPSV`, and `xHPSV` as one-shot convenience APIs, followed by
their `xPPSVX`, `xSPSVX`, and `xHPSVX` expert counterparts. The selected
binding crate also exposes the complex-symmetric `CSPSV`/`ZSPSV` and
`CSPSVX`/`ZSPSVX` variants.

Equilibration (`xPPEQU`), condition estimation (`xPPCON`, `xSPCON`, `xHPCON`), refinement
(`xPPRFS`, `xSPRFS`, `xHPRFS`), and packed norms (`xLANSP`, `xLANHP`)
and packed rank updates (`xSPR`, `xSPR2`, `xHPR`, `xHPR2`) are already implemented.

### Priority 2: low-level reductions and interoperability

The high-level basic, divide-and-conquer, selected, and generalized packed
eigensolvers are complete. Remaining expert building blocks include
`xSPTRD`/`xHPTRD`, transformation generation/application, and
`xSPGST`/`xHPGST`. Traditional-packed/full/RFP conversions are optional
interoperability work.

Most users should continue to use the existing high-level eigensolver APIs.

---

## Condensed completeness table

| Matrix type | Implemented families | Major missing families |
|---|---|---|
| Lower/upper triangular | `TPMV`, `TPSV`, `TPTRS`, `TPTRI`, `TPCON`, `TPRFS`, `LANTP` | `LATPS` (unsupported by the selected Rust `lapack` crate); mostly packed/full/RFP conversions |
| Real symmetric | `SPMV`, `SPR/2`, `SPTRF`, `SPTRS`, `SPTRI`, `SPCON`, `SPRFS`, `LANSP`, `SPEV/D/X`, `SPGV/D/X` | `SPSV/X`; low-level reductions |
| Complex symmetric | `SPTRF`, `SPTRS`, `SPTRI`, `SPCON`, `SPRFS`, `LANSP` | `SPSV/X`; Hermitian eigensolvers are not applicable |
| SPD / HPD | `SPMV`/`HPMV`, `PPTRF`, `PPTRS`, `PPTRI`, `PPCON`, `PPEQU`, `PPRFS`, `LANSP`/`LANHP`, symmetric/Hermitian `PEV/D/X` and `PGV/D/X` | `PPSV/X`; unrestricted updates require conversion to symmetric/Hermitian |
| Hermitian | `HPMV`, `HPR/2`, `HPTRF`, `HPTRS`, `HPTRI`, `HPCON`, `HPRFS`, `LANHP`, `HPEV/D/X`, `HPGV/D/X` | `HPSV/X`; low-level reductions |

## Maintenance note

When a family is implemented, update this document only after all of the following exist:

1. scalar backend dispatch for every valid scalar family;
2. a safe public API with the appropriate owned/view/view-mut allocation policy;
3. numerical examples or tests covering each supported scalar type;
4. documentation of destructive versus non-destructive behavior.
