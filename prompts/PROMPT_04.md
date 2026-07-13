The following tasks apply to:

https://github.com/evnekdev/matrixpacked

## Common workflow contract

Apply this contract to every numbered task:

1. Fetch the remote repository.
2. Switch to `master`.
3. Pull with fast-forward only.
4. Confirm the previous PR has already been merged.
5. Create the branch specified in the task.
6. Do not modify `master` directly.
7. Use only functions exposed by the Rust `lapack` and `blas` crates.
8. Do not add handwritten `extern "C"` or direct Fortran-symbol bindings.
9. Inspect the exact signatures in the installed binding-crate source before coding.
10. Preserve true traditional packed storage; never expand to dense storage merely to call a routine.
11. Support owned, immutable-view, and mutable-view storage intelligently:

    * owned values should reuse allocations where destructive LAPACK operations permit;
    * mutable views should support zero-copy destructive operations;
    * immutable views may allocate only packed storage;
    * BLAS rank updates require mutable storage.
12. Add examples and tests for every valid scalar family.
13. Update:

    * `PACKED_LAPACK_FUNCTIONS.md`
    * `EXAMPLE_COVERAGE.md`
    * `README.md` where the feature is user-facing.
14. Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo check --all-targets
cargo check --examples
cargo test
git diff --check
```

15. Inspect:

```bash
git diff master...HEAD --stat
git diff master...HEAD
```

16. Open a draft PR.
17. Do not continue changing the branch after declaring it ready.
18. End only when no more commits are planned, using this exact standalone line:

**Safe to rebase and merge.**

---

# 1. Reconcile LAPACK status and example coverage

Branch:

```text
agent/reconcile-packed-api-status
```

Commit and PR title:

```text
Reconcile packed API status
```

## Goal

Audit the repository’s actual public API and make its status and coverage documentation accurately match the code.

This is a documentation-and-audit PR. Do not implement new numerical functionality in it.

## Required audit

Search the complete repository for all packed routine families currently documented:

```bash
git grep -n -E \
'TPMV|TPSV|TPTRS|TPTRI|TPCON|TPRFS|LANTP|LATPS|\
SPMV|SPR2?|SPTRF|SPTRS|SPTRI|SPCON|SPRFS|LANSP|\
PPTRF|PPTRS|PPTRI|PPCON|PPEQU|PPRFS|\
HPMV|HPR2?|HPTRF|HPTRS|HPTRI|HPCON|HPRFS|LANHP|\
SPEV|SPEVD|SPEVX|HPEV|HPEVD|HPEVX|\
SPGV|SPGVD|SPGVX|HPGV|HPGVD|HPGVX'
```

Inspect at minimum:

```text
src/backend.rs
src/norms.rs
src/condition.rs
src/refinement.rs
src/triangular.rs
src/factorization.rs
src/eigen.rs
src/eigen_divide_conquer.rs
src/eigen_selected.rs
src/generalized_eigen.rs
src/generalized_eigen_divide_conquer.rs
src/generalized_eigen_selected.rs
PACKED_LAPACK_FUNCTIONS.md
EXAMPLE_COVERAGE.md
README.md
examples/
tests/
```

Use actual source implementations, public methods, tests, and examples as the source of truth.

## Known inconsistencies to investigate

The status document may still incorrectly mark these as missing despite implementation:

```text
xLANSP
xLANHP
xSPEV
xSPEVD
xSPEVX
xHPEV
xHPEVD
xHPEVX
xSPGV
xSPGVD
xSPGVX
xHPGV
xHPGVD
xHPGVX
```

Verify, do not blindly change them.

Check whether complex-symmetric:

```text
CSPCON
ZSPCON
CSPRFS
ZSPRFS
```

are implemented and publicly usable. The detailed and condensed status sections may disagree.

## Documentation requirements

For every routine family, distinguish:

* implemented for all valid scalars;
* implemented for only specific scalars;
* backend only;
* missing but exposed by the Rust binding crate;
* unsupported because the selected Rust binding crate does not expose it;
* not applicable to the matrix structure.

Correct the condensed completeness table.

Correct the recommended implementation order so it does not recommend already completed eigensolver work.

Calculate the example count from the files rather than manually guessing it.

Suggested command:

```bash
find examples -maxdepth 1 -type f -name '*.rs' ! -name 'common.rs' | wc -l
```

Ensure `EXAMPLE_COVERAGE.md` matches the real examples.

## Validation

Run repository-wide checks for contradictory status entries.

For example, every `**Missing**` family should be searched in `src/`, `examples/`, and `tests/` before being retained as missing.

Include an audit summary in the PR description listing:

* statuses corrected;
* stale roadmap sections corrected;
* genuine missing families remaining;
* unsupported families;
* example count.

---

# 2. Implement packed symmetric and Hermitian rank updates

Branch:

```text
agent/packed-rank-updates
```

Commit and PR title:

```text
Implement packed rank updates
```

## Goal

Implement BLAS packed-storage rank-update routines:

```text
SSPR
DSPR
SSPR2
DSPR2
CHPR
ZHPR
CHPR2
ZHPR2
```

These are BLAS operations, not LAPACK drivers.

Do not implement nonexistent complex-symmetric `CSPR` or `ZSPR` APIs. Complex packed updates must use Hermitian semantics through `HPR`/`HPR2`.

## Matrix types

Implement:

### Real symmetric

For:

```rust
PackedSymmetric<f32, S>
PackedSymmetric<f64, S>
```

Operations:

```rust
rank1_update_in_place(alpha, x)
rank1_update_strided_in_place(alpha, x, incx)
rank2_update_in_place(alpha, x, y)
rank2_update_strided_in_place(alpha, x, incx, y, incy)
```

Semantics:

```text
A := A + alpha * x * x^T
A := A + alpha * x * y^T + alpha * y * x^T
```

### Complex Hermitian

For:

```rust
PackedHermitian<Complex32, S>
PackedHermitian<Complex64, S>
```

Operations:

```rust
rank1_update_in_place(alpha_real, x)
rank1_update_strided_in_place(alpha_real, x, incx)
rank2_update_in_place(alpha_complex, x, y)
rank2_update_strided_in_place(alpha_complex, x, incx, y, incy)
```

Semantics:

```text
A := A + alpha * x * x^H
A := A + alpha * x * y^H + conj(alpha) * y * x^H
```

For `HPR`, `alpha` must be the associated real type.

## Storage requirements

Rank updates mutate the packed matrix. Expose them only when:

```rust
S: PackedStorageMut<T>
```

Do not clone implicitly in the core update methods.

Optional allocating convenience methods may be added only if clearly named, such as:

```rust
updated_rank1(...)
updated_rank2(...)
```

but they are not required.

## Backend traits

Add narrowly scoped internal traits, for example:

```rust
trait RealSymmetricPackedRankUpdate
trait HermitianPackedRankUpdate
```

Do not overload the factorization backend traits with unrelated BLAS operations if separate traits are clearer.

Dispatch exclusively through the Rust `blas` crate.

## Strided-vector validation

Reuse or generalize existing packed-triangular strided-vector validation.

Correctly support positive and negative increments if the Rust BLAS binding and project conventions permit them.

Validate:

* nonzero increment;
* required physical slice length;
* matrix dimension;
* vector dimensions;
* conversion to BLAS integer types.

No panic on invalid input.

## SPD/HPD policy

Do not expose unrestricted rank updates directly as mutating methods on `PackedSPD<T>` without addressing the invariant.

A generic negative rank update may destroy positive definiteness.

Choose one of these safe designs:

### Preferred design

Perform the operation but return the less restrictive structure:

```rust
PackedSPD<T>::into_symmetric()
PackedSPD<Complex<_>>::into_hermitian()
```

then update that type.

### Alternative

Expose restricted SPD/HPD rank-1 additions only:

```rust
positive_rank1_update_in_place(alpha, x)
```

with real `alpha >= 0`.

Do not claim that `HPR2` or arbitrary `SPR2` preserves positive definiteness.

Document the decision clearly.

## Examples

Add at least:

```text
examples/blas_symmetric_f32_spr.rs
examples/blas_symmetric_f64_spr.rs
examples/blas_symmetric_f32_spr2.rs
examples/blas_symmetric_f64_spr2.rs

examples/blas_hermitian_c32_hpr.rs
examples/blas_hermitian_c64_hpr.rs
examples/blas_hermitian_c32_hpr2.rs
examples/blas_hermitian_c64_hpr2.rs
```

Follow the repository’s naming convention if all existing numerical examples use the `lapack_` prefix, but note in documentation that these are BLAS routines.

Each example must:

1. create a small packed matrix;
2. apply the update;
3. compare every logical matrix entry against an explicitly computed expected matrix;
4. verify conjugation for Hermitian off-diagonal entries;
5. verify the diagonal remains real after `HPR` and `HPR2`.

## Tests

Cover:

* owned storage;
* mutable views;
* upper/lower logical access if the crate has both orientations;
* stride greater than one;
* invalid zero stride;
* insufficient slice length;
* real rank-1 and rank-2;
* complex Hermitian rank-1 and rank-2;
* zero coefficient;
* one-dimensional and zero-dimensional matrices;
* SPD/HPD invariant policy.

Do not test only packed bytes; test logical `(i, j)` access as well.

---

# 3. Implement positive-definite packed equilibration

Branch:

```text
agent/packed-equilibration
```

Commit and PR title:

```text
Implement packed equilibration
```

## Goal

Implement the packed positive-definite equilibration family:

```text
SPPEQU
DPPEQU
CPPEQU
ZPPEQU
```

Use only functions exposed by the Rust `lapack` crate.

## Intended API

Introduce a result type similar to:

```rust
pub struct Equilibration<R> {
    pub scaling: Vec<R>,
    pub condition_ratio: R,
    pub maximum_diagonal: R,
}
```

Use terminology consistent with the actual LAPACK outputs.

Expose:

```rust
PackedSPD<T, S>::equilibration()
```

or:

```rust
PackedSPD<T, S>::equilibration_factors()
```

The method must not modify the matrix.

Optionally expose:

```rust
equilibrate_in_place(&mut self, factors: &Equilibration<_>)
```

but only if its semantics are carefully defined and implemented without dense expansion.

## Correct mathematics

Determine from the exact LAPACK documentation and binding signature:

* how scaling factors are computed;
* how the scaling condition indicator is interpreted;
* how the largest diagonal entry is reported;
* what happens when a diagonal entry is non-positive;
* how complex HPD diagonals are treated.

Map LAPACK’s failure index to a meaningful `PackedMatrixError`.

For complex Hermitian positive-definite matrices, ensure diagonal imaginary components are treated according to LAPACK semantics and crate invariants.

## Application helper

If adding an application method, scale packed entries directly:

```text
A(i,j) := s[i] * A(i,j) * s[j]
```

For Hermitian storage this preserves conjugate symmetry when done correctly.

Avoid repeated logical indexing if a single packed-storage traversal is more efficient.

Provide a checked helper for applying externally supplied scaling factors:

```rust
apply_equilibration_in_place(&mut self, scaling: &[T::Real])
```

Validate `scaling.len() == n`.

## Examples

Add:

```text
examples/lapack_spd_f32_ppequ.rs
examples/lapack_spd_f64_ppequ.rs
examples/lapack_spd_c32_ppequ.rs
examples/lapack_spd_c64_ppequ.rs
```

Each example must verify:

* scaling-vector length;
* finite, positive scaling factors;
* returned condition indicator;
* scaled diagonal values;
* complex HPD behavior.

## Tests

Cover:

* already well-scaled matrix;
* strongly unbalanced diagonal;
* non-positive diagonal failure;
* real and complex;
* owned and immutable view;
* empty and `1 × 1` cases;
* application of scaling to mutable packed storage.

Update the status table from `Missing` to the exact implemented scalar set.


Apply the common workflow contract from Prompt 1 to each task below.

# 4. Implement simple packed solve drivers

Branch:

```text
agent/packed-simple-solve-drivers
```

Commit and PR title:

```text
Implement packed solve drivers
```

## Goal

Implement one-shot factor-and-solve drivers where exposed by Rust `lapack`:

```text
xPPSV
xSPSV
xHPSV
```

Investigate whether the binding crate exposes complex-symmetric:

```text
CSPSV
ZSPSV
```

Implement them only when the binding exists and the existing `PackedSymmetric<Complex<_>>` factorization semantics match it.

## Why these APIs exist

The crate already offers reusable factorization plus solve. These drivers are convenience APIs for callers who need only one solve and do not need to retain the factorization.

Do not replace the existing reusable APIs.

## Public APIs

Preferred borrowing API:

```rust
matrix.solve_once(&b, nrhs)
```

This should:

1. clone only packed matrix storage;
2. clone the right-hand-side buffer;
3. call the one-shot LAPACK driver;
4. return the solution;
5. leave the original matrix and RHS unchanged.

Preferred consuming API:

```rust
matrix.into_solve_once(b, nrhs)
```

This should reuse the owned packed allocation.

For mutable packed views, consider:

```rust
solve_once_in_place(&mut self, b: &mut [T], nrhs: usize)
```

Document that the matrix is overwritten by its factorization.

## Families

### Positive definite

Support:

```text
SPPSV
DPPSV
CPPSV
ZPPSV
```

### Real symmetric indefinite

Support:

```text
SSPSV
DSPSV
```

### Complex Hermitian indefinite

Support:

```text
CHPSV
ZHPSV
```

### Complex symmetric

Support `CSPSV`/`ZSPSV` only if present in the selected Rust binding and compatible with current factorization code.

## Return values

For indefinite drivers, expose pivot information only if it provides real user value.

A possible consuming result is:

```rust
pub struct PackedSolveResult<T> {
    pub solution: Vec<T>,
}
```

Do not add a result wrapper merely to contain one vector.

Prefer returning:

```rust
Result<Vec<T>, PackedMatrixError>
```

unless factorization metadata must also be preserved.

## Validation

Check:

* RHS length equals `n * nrhs`;
* column-major RHS layout;
* integer conversions;
* zero RHS count;
* singular-factorization errors;
* positive-definiteness failures;
* mutable-view overwrite behavior.

## Examples

Add one example per valid scalar family and driver.

Examples should compare one-shot driver results against:

```rust
matrix.factorize()?.solve(...)
```

to prove consistency.

## Tests

Cover:

* one and multiple RHS;
* borrowing and consuming APIs;
* mutable view;
* invalid RHS length;
* SPD/HPD failure;
* singular indefinite matrix;
* real and complex.

Document that users who solve repeatedly should retain a factorization instead.

---

# 5. Implement expert packed solve drivers

Branch:

```text
agent/packed-expert-solve-drivers
```

Commit and PR title:

```text
Implement packed expert solve drivers
```

## Goal

Implement the expert packed solve drivers exposed by Rust `lapack`:

```text
xPPSVX
xSPSVX
xHPSVX
```

Conditionally include complex-symmetric `CSPSVX`/`ZSPSVX` only if the selected binding crate exposes them and their semantics match the crate.

These drivers combine factorization, optional reuse of supplied factors, condition estimation, refinement, error bounds, and—where applicable—equilibration.

## Do not expose raw LAPACK flags directly

Create typed options.

Possible structure:

```rust
pub enum FactorUsage {
    Compute,
    UseProvided,
}

pub enum EquilibrationMode {
    None,
    Compute,
    Provided,
}

pub struct ExpertSolveOptions<R> {
    pub factor_usage: FactorUsage,
    pub equilibration: EquilibrationMode,
    pub scaling: Option<Vec<R>>,
}
```

Adapt this to the exact driver semantics. Do not invent options unsupported by a given routine.

## Result type

Introduce one coherent result type:

```rust
pub struct ExpertSolveResult<T, R> {
    pub solution: Vec<T>,
    pub reciprocal_condition_number: R,
    pub forward_error: Vec<R>,
    pub backward_error: Vec<R>,
    pub equilibration: Option<Equilibration<R>>,
}
```

Include factorization or pivot data only if the API intentionally supports factor reuse.

Avoid exposing internal workspace.

## Architecture

Do not duplicate the logic already present in:

* condition estimation;
* iterative refinement;
* equilibration;
* factorization and solve.

Reuse public or internal validation helpers where possible.

However, call the actual expert LAPACK driver rather than manually composing separate routines under an API named `*SVX`.

## Supplied-factor workflow

If supporting `FACT='F'`, design a separate advanced API or builder requiring all necessary inputs:

* supplied factorization;
* pivots where relevant;
* scaling factors where relevant;
* equilibration state.

Do not allow logically inconsistent combinations to reach LAPACK.

A simpler first PR may support only LAPACK-computed factorization, but then document the reduced scope accurately. Do not mark the complete family implemented if major driver modes are omitted.

## Error mapping

Map distinctly:

* illegal argument;
* singular factorization;
* not-positive-definite leading minor;
* tiny reciprocal condition number;
* refinement/convergence information;
* invalid supplied factors or scaling.

Do not treat a very ill-conditioned but successfully solved system as an ordinary hard error unless LAPACK specifies failure.

## Examples

For each matrix family, include:

1. a well-conditioned system;
2. an ill-scaled or ill-conditioned system;
3. verification of solution residual;
4. verification of result-vector lengths;
5. inspection of `rcond`, `ferr`, and `berr`.

Cover all valid scalar families.

## Tests

Include:

* one/multiple RHS;
* real/complex;
* equilibration on/off;
* invalid options;
* non-positive-definite `B` or `A` where relevant;
* singular indefinite systems;
* factor reuse if implemented;
* agreement with factorization + refinement APIs.

README guidance should explain the choice:

```text
simple solve -> reusable factorization -> expert solve
```

and when each is appropriate.

---

# 6. Implement packed tridiagonal reductions and transformations

Branch:

```text
agent/packed-tridiagonal-reductions
```

Commit and PR title:

```text
Implement packed tridiagonal reductions
```

## Goal

Implement low-level packed eigensolver building blocks:

### Real symmetric

```text
SSPTRD
DSPTRD
SOPGTR
DOPGTR
SOPMTR
DOPMTR
```

### Complex Hermitian

```text
CHPTRD
ZHPTRD
CUPGTR
ZUPGTR
CUPMTR
ZUPMTR
```

Implement only functions available through Rust `lapack`.

## Scope distinction

These are lower-level expert APIs. Existing high-level packed eigensolvers should remain the normal path.

Do not reimplement existing `SPEV/HPEV` methods using the new public APIs unless there is a compelling internal simplification with no regression.

## Reduction result types

Introduce structured types representing the overwritten packed matrix and reflector data.

Possible design:

```rust
pub struct SymmetricPackedTridiagonal<T, S> {
    packed_reflectors: PackedSymmetric<T, S>,
    diagonal: Vec<T>,
    off_diagonal: Vec<T>,
    tau: Vec<T>,
}

pub struct HermitianPackedTridiagonal<T, S> {
    packed_reflectors: PackedHermitian<T, S>,
    diagonal: Vec<T::Real>,
    off_diagonal: Vec<T::Real>,
    tau: Vec<T>,
}
```

Use names appropriate to existing conventions.

Do not expose fields publicly unless direct access is useful. Provide accessors.

## APIs

Borrowing:

```rust
matrix.tridiagonal_reduction()
```

Consuming:

```rust
matrix.into_tridiagonal_reduction()
```

Mutable-view destructive:

```rust
matrix.tridiagonal_reduction_in_place()
```

The consuming and mutable APIs should avoid copying packed storage.

## Generate Q

Expose:

```rust
reduction.generate_q()
```

This returns an owned dense orthogonal/unitary matrix because `OPGTR/UPGTR` outputs full `Q`.

Document its column-major layout.

Do not pretend `Q` remains packed.

## Apply Q

Expose typed APIs around `OPMTR/UPMTR`:

```rust
reduction.apply_q_left_in_place(...)
reduction.apply_q_right_in_place(...)
```

Support:

* left/right application;
* transpose for real;
* conjugate transpose for complex;
* column-major dense target matrices;
* explicit row/column dimensions and leading dimension validation.

Use enums rather than raw LAPACK characters.

## Correctness tests

Verify the decomposition:

### Real

```text
A ≈ Q T Q^T
```

### Complex

```text
A ≈ Q T Q^H
```

Also verify:

* `Q^T Q ≈ I` or `Q^H Q ≈ I`;
* `apply_q_*` matches explicit multiplication;
* upper and lower packed storage;
* empty, `1 × 1`, and small nontrivial matrices;
* all valid scalar types.

## Examples

Add individual examples for:

* `SPTRD`;
* `HPTRD`;
* `OPGTR`;
* `UPGTR`;
* left/right `OPMTR`;
* left/right `UPMTR`;

with scalar coverage distributed across examples.

Document that these APIs are for users who need the intermediate tridiagonal form or reflector representation.


Apply the common workflow contract from Prompt 1 to each task below.

# 7. Implement generalized packed reductions

Branch:

```text
agent/packed-generalized-reductions
```

Commit and PR title:

```text
Implement packed generalized reductions
```

## Goal

Implement:

```text
SSPGST
DSPGST
CHPGST
ZHPGST
```

These routines reduce a generalized symmetric/Hermitian-definite packed eigenproblem to standard form using a previously factorized SPD/HPD matrix `B`.

## Required preconditions

The API must require:

* symmetric real or Hermitian complex packed `A`;
* packed Cholesky factorization of positive-definite `B`;
* equal dimensions;
* compatible scalar types;
* a typed generalized eigenproblem variant corresponding to LAPACK `ITYPE`.

Reuse the existing enum introduced for generalized eigensolvers, if one exists.

Do not introduce a second incompatible enum.

## Intended API

Possible pattern:

```rust
a.reduce_generalized_in_place(&b_factor, problem_type)
```

and:

```rust
a.generalized_reduction(&b_factor, problem_type)
```

The in-place method should overwrite `A` with the reduced standard-form matrix.

The borrowing method should clone only packed `A`.

The Cholesky factor of `B` should be borrowed and unchanged.

## Mathematical documentation

Document the transformation for each problem type:

```text
A*x = lambda*B*x
A*B*x = lambda*x
B*A*x = lambda*x
```

Explain what reduced matrix is produced and how eigenvectors must be transformed back.

Avoid vague wording such as “converts generalized to standard” without describing the role of the Cholesky factor.

## Integration

Review existing generalized eigenvalue implementations.

Factor shared validation into helpers where useful.

Do not alter the existing high-level generalized eigensolver results unless fixing a concrete defect.

## Examples and tests

For real and complex examples:

1. form `A` and positive-definite `B`;
2. factorize `B`;
3. reduce `A`;
4. solve the resulting standard problem using existing packed eigensolvers;
5. compare eigenvalues with the existing high-level generalized eigensolver;
6. verify generalized residuals.

Cover all three problem types, upper/lower storage, real/complex, and dimension mismatch.

---

# 8. Implement traditional packed conversion interoperability

Branch:

```text
agent/packed-format-conversions
```

Commit and PR title:

```text
Implement packed format conversions
```

## Goal

Implement packed-format conversion routines exposed by Rust `lapack`, including as available:

```text
xTPTTR
xTRTTP
xTPTTF
xTFTTP
```

Before coding, verify exactly which scalar variants the selected Rust `lapack` crate exposes.

Do not add manual FFI for missing variants.

## Separate the formats clearly

The crate already uses traditional packed storage.

Define clearly:

* traditional packed (`TP`);
* full triangular (`TR`);
* rectangular full packed (`TF`/RFP).

Do not call a full dense triangular matrix “packed”.

## Full triangular conversion API

Traditional packed to full:

```rust
packed.to_full_triangular()
```

Return a deliberately named dense triangular representation.

Do not introduce a major new general matrix abstraction merely for this feature.

A reasonable minimal result is:

```rust
pub struct FullTriangular<T> {
    pub data: Vec<T>,
    pub dimension: usize,
    pub triangle: Triangle,
}
```

If the project already interoperates with nalgebra, optionally provide feature-gated conversion to `DMatrix<T>`.

Keep the LAPACK-backed core independent of nalgebra unless nalgebra is already a mandatory dependency.

Full triangular to packed:

```rust
PackedLower::from_full_triangular(...)
PackedUpper::from_full_triangular(...)
```

Validate dimensions and triangle selection.

## RFP representation

Introduce an explicit RFP type only if all layout metadata is represented safely:

```rust
pub struct RectangularFullPacked<T, S> {
    storage: S,
    dimension: usize,
    triangle: Triangle,
    transr: RfpTranspose,
}
```

Account for:

* odd/even matrix dimension;
* normal/transposed RFP representation;
* upper/lower triangle;
* real/complex scalar behavior.

Do not expose raw `TRANSR` characters.

## Round-trip tests

Verify:

```text
TP -> TR -> TP
TP -> RFP -> TP
```

for:

* lower and upper;
* odd and even dimensions;
* `f32`, `f64`, `Complex32`, `Complex64`;
* empty and `1 × 1` matrices.

Compare logical matrix entries and exact packed storage where round trips should be exact.

## Examples

Create focused examples for each conversion family.

Update the interoperability section of the README.

Document that RFP is a distinct storage format intended to retain compact storage while enabling different LAPACK kernels; do not imply that arbitrary packed matrix operations automatically work on it.

If the selected Rust bindings lack these conversion routines, stop that specific subfamily, document it as unsupported by the selected binding crate, and do not add direct FFI.

---

# 9. Add derived factorization diagnostics

Branch:

```text
agent/packed-factor-diagnostics
```

Commit and PR title:

```text
Add packed factor diagnostics
```

## Goal

Add useful high-level diagnostics that can be derived from existing packed factorizations without needing new LAPACK routines:

* determinant;
* sign plus log-absolute-determinant;
* log-determinant for SPD/HPD;
* inertia for real symmetric and complex Hermitian indefinite factorizations;
* singularity diagnostics where meaningful.

This is not a direct missing LAPACK binding, but it fills an important public-API gap identified in the packed-function roadmap.

## SPD/HPD log determinant

For a Cholesky factor:

```text
log(det(A)) = 2 * sum(log(real(diag(L))))
```

or the corresponding upper-factor formula.

Expose:

```rust
factor.log_determinant()
factor.determinant()
```

For valid SPD/HPD matrices, the determinant is real and positive.

Return the associated real scalar type.

Use a numerically stable log form as the primary implementation.

The direct determinant may overflow or underflow; document this.

## Triangular determinant

For non-unit diagonal:

```text
det(A) = product(diagonal)
```

For unit diagonal:

```text
det(A) = 1
```

Expose determinant and log-absolute-determinant where scalar semantics make sense.

For complex triangular matrices, use a complex determinant and consider returning magnitude/phase diagnostics rather than forcing a real sign.

## Symmetric/Hermitian indefinite factors

Inspect the pivot encoding produced by packed Bunch–Kaufman factorization.

Correctly handle:

* `1 × 1` diagonal blocks;
* `2 × 2` diagonal blocks;
* upper/lower pivot conventions;
* real symmetric factors;
* complex Hermitian factors.

Implement:

```rust
factor.slogdet()
factor.inertia()
```

Possible result:

```rust
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

pub struct SignedLogDet<R> {
    pub sign: R,
    pub log_abs: R,
}
```

For complex Hermitian matrices, the determinant is real, but roundoff handling must be careful.

For complex symmetric matrices, determinant phase is generally complex; do not reuse Hermitian APIs incorrectly.

## Numerical thresholds

Do not silently classify small values as zero with an undocumented arbitrary constant.

Either:

* determine exact zero from factor blocks;
* accept a user-supplied tolerance;
* provide both exact and tolerance-aware inertia methods.

Document the choice.

## Tests

Validate against analytically known matrices:

* positive definite;
* negative definite;
* indefinite;
* singular;
* `2 × 2` pivot blocks;
* triangular unit/non-unit diagonal;
* real and complex;
* determinant sign and log magnitude;
* overflow-resistant log determinant.

Do not calculate expected results using the methods under test.

## Documentation

Explain why:

* solving is preferable to explicitly forming an inverse;
* log determinant is preferable to a direct product;
* inertia is available from the factorization without eigenvalue decomposition.

Keep this PR separate from new native bindings.
