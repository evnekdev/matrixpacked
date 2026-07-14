# Prompt 04 — Add structural validation and configurable tolerances

Work on:

https://github.com/evnekdev/matrixpacked

## Starting conditions

Start after Prompt 03 is merged.

Create:

```text
agent/nalgebra-conversion-validation
```

## Goal

Add strict, tolerance-aware conversions from nalgebra full matrices into packed matrix types.

These APIs must distinguish validation from intentional triangle extraction.

## Public tolerance type

Design a reusable public type under the `nalgebra-interop` feature.

Possible API:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConversionTolerance<R> {
    pub absolute: R,
    pub relative: R,
}
```

Add constructors:

```rust
pub const fn new(absolute: R, relative: R) -> Self;
```

and well-documented defaults where appropriate.

Possible trait:

```rust
pub trait DefaultConversionTolerance {
    fn default_conversion_tolerance() -> ConversionTolerance<Self>;
}
```

Keep the API small.

Do not create an overly generic approximate-equality framework for the whole crate.

## Comparison rule

Use:

```text
|a - b| <= absolute + relative * max(|a|, |b|)
```

For complex scalars use complex magnitude.

Validate that tolerance values are:

- finite where the scalar supports this check;
- nonnegative.

Return an error for invalid tolerance values.

Do not silently take absolute values of negative tolerances.

## Strict triangular conversions

Implement:

```rust
PackedLower::try_from_dmatrix(&matrix, tolerance)
PackedUpper::try_from_dmatrix(&matrix, tolerance)
```

Requirements:

- matrix must be square;
- opposite triangle must be approximately zero;
- selected triangle is packed;
- diagonals are unrestricted unless the type has another contract.

Report the first offending coordinate, or provide enough diagnostic detail to identify the structural violation.

## Strict symmetric conversion

Implement:

```rust
PackedSymmetric::try_from_dmatrix(&matrix, tolerance)
```

Requirements:

```text
A(i,j) ≈ A(j,i)
```

For complex scalars, do not conjugate.

Decide which triangle is stored after validation.

Use the crate’s existing storage convention.

## Strict Hermitian conversion

Implement:

```rust
PackedHermitian::try_from_dmatrix(&matrix, tolerance)
```

Requirements:

```text
A(i,j) ≈ conj(A(j,i))
imag(A(i,i)) ≈ 0
```

Store a real-valued diagonal according to the crate’s invariant.

Do not preserve arbitrary imaginary diagonal noise.

Document whether accepted small imaginary parts are normalized to zero.

## Strict SPD/HPD conversion

Implement two levels.

### Structural validation only

```rust
PackedSPD::try_from_structured_dmatrix(&matrix, tolerance)
```

This validates symmetry/Hermitian structure but does not prove positive definiteness.

Choose a clearer name if available.

### Full positive-definiteness validation

```rust
PackedSPD::try_from_dmatrix(&matrix, tolerance)
```

This must:

1. validate square shape;
2. validate symmetry or Hermitian structure;
3. verify positive definiteness using nalgebra Cholesky or another nalgebra full-storage factorization;
4. return owned packed storage only on success.

Do not call matrixpacked’s LAPACK factorization as the validator; the purpose of this conversion is nalgebra interoperability and independent validation.

Do not explicitly compute eigenvalues merely to test positive definiteness if Cholesky is available.

## Error model

Add meaningful `PackedMatrixError` variants, or a feature-gated conversion-specific error type if that is cleaner.

Potential conditions:

```text
NonSquareMatrix
NotTriangular
NotSymmetric
NotHermitian
NonRealHermitianDiagonal
NotPositiveDefinite
InvalidTolerance
```

Include useful fields:

- offending row;
- offending column;
- actual values or magnitudes where practical;
- rows and columns for shape errors.

Be mindful that including generic scalar values in the crate error enum may complicate its type.

A non-generic error can store coordinates and textual structure names without storing values.

Avoid a large API redesign of the existing error type.

## Canonicalization policy

After approximate validation, choose a canonical stored value.

For symmetric/Hermitian pairs, options include:

1. store the selected triangle exactly;
2. average the pair;
3. prefer lower or upper.

Preferred policy:

- store the crate’s selected physical triangle exactly;
- treat the opposite triangle only as validation evidence.

Do not average values unless clearly documented; averaging changes user data.

For Hermitian diagonal, normalize accepted tiny imaginary parts to zero if required by the invariant.

## Tests

Add:

```text
tests/nalgebra_conversion_validation.rs
```

Required tests:

### Triangular

- exact triangular accepted;
- small opposite-triangle noise accepted;
- excessive noise rejected;
- lower and upper;
- real and complex.

### Symmetric

- exact accepted;
- tolerance-level mismatch accepted;
- larger mismatch rejected;
- complex symmetric without conjugation.

### Hermitian

- exact accepted;
- conjugate mismatch rejected;
- tiny imaginary diagonal accepted and normalized;
- excessive imaginary diagonal rejected.

### SPD/HPD

- valid SPD accepted;
- valid HPD accepted;
- symmetric but indefinite rejected;
- Hermitian but indefinite rejected;
- semidefinite rejected;
- malformed structure rejected before Cholesky;
- badly scaled but valid positive-definite matrix.

### Tolerance

- zero tolerance;
- invalid negative tolerance;
- NaN tolerance if representable;
- relative versus absolute behavior.

## Documentation

Clearly distinguish:

```text
from_*_triangle()
```

as extraction and:

```text
try_from_dmatrix()
```

as structural validation.

Document the cost of SPD validation.

## Validation and PR

Commit:

```text
Validate nalgebra matrix conversions
```

Branch:

```text
agent/nalgebra-conversion-validation
```

PR title:

```text
Validate nalgebra matrix conversions
```

Finish only after complete with:

**Safe to rebase and merge.**
