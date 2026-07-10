# Supported operations

This crate intentionally keeps BLAS/LAPACK packed storage as the computational representation. It does not expand matrices into a rectangular dense allocation.

## Nalgebra-like operations reproduced

| Operation | Lower/Upper | Symmetric | SPD/HPD | Hermitian |
|---|---:|---:|---:|---:|
| `+`, `+=` | yes | yes | yes (sum remains PD) | yes |
| `-`, `-=`, unary `-` | yes | yes | deliberately not typed as SPD | yes |
| scalar `*`, `/`, assignments | yes | yes | deliberately omitted unless definiteness can be preserved | deliberately omitted for complex scalars |
| `component_mul`, `component_div` | yes | yes | omitted because the result is not generally guaranteed PD | planned |
| matrix-vector `*` | `xTPMV` | `xSPMV` for real | `xSPMV`/`xHPMV` | `xHPMV` |
| solve vector | `xTPTRS` | packed Bunch-Kaufman | packed Cholesky | packed Hermitian Bunch-Kaufman |
| inverse in place | `xTPTRI` | factor then `xSPTRI` | factor then `xPPTRI` | factor then `xHPTRI` |

Traditional packed BLAS has no packed Level-3 matrix-matrix multiplication routine. A general `A * B` operator is therefore intentionally absent: silently expanding to a full matrix would defeat the crate's purpose.

## Allocation policy

- `View`: multiplication can write into caller-provided output without allocation. Factorization clones because LAPACK overwrites its matrix.
- `Owned`: `*_in_place(self)` consumes and reuses the existing `Vec` as factor storage.
- `ViewMut`: `*_in_place(self)` consumes the view and factors the borrowed packed slice directly; only pivot/work arrays required by LAPACK are allocated.
- Triangular solve never overwrites the matrix, so all storage forms solve without copying the matrix.
