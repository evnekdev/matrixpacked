# LAPACK example coverage

The `examples/` directory contains **188** checked Rust examples (excluding the
shared `examples/common.rs` module). This count is derived from the files:

```bash
find examples -maxdepth 1 -type f -name '*.rs' ! -name 'common.rs' | wc -l
```

For an exact inventory, use:

```bash
find examples -maxdepth 1 -type f -name '*.rs' ! -name 'common.rs' -printf '%f\n' | sort
```

Run one numerical example with:

```bash
cargo run --example lapack_lower_f64_tpmv --features openblas-static
```

Run every `lapack_` example on a Unix-like shell with:

```bash
for example in $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].targets[] | select(.kind[] == "example") | .name' | grep '^lapack_'); do
  cargo run --quiet --example "$example" --features openblas-static || exit 1
done
```

## Coverage by family

The triangular examples cover lower and upper storage for
`f32`/`f64`/`Complex32`/`Complex64` across `TPMV`, `TPSV`, `TPTRS`, `TPTRI`,
`TPCON`, `TPRFS`, and `LANTP`.

Positive-definite examples cover all four scalar families for packed
matrix-vector multiplication, `PPTRF`, `PPTRS`, `PPTRI`, `PPCON`, `PPEQU`,
and `PPRFS`. The equilibration examples check the returned scaling diagnostics
and directly scaled real and complex HPD diagonals.

One-shot solve examples cover `SPPSV`/`DPPSV`/`CPPSV`/`ZPPSV`, real and
complex-symmetric `xSPSV`, and `CHPSV`/`ZHPSV`. Each agrees with the matching
reusable factorization-and-solve API.

Expert solve examples cover computed-factor `xPPSVX`, `xSPSVX`, and `xHPSVX`
for every valid scalar, including condition/error outputs and equilibration.

Low-level reduction examples cover `xSPTRD`/`xHPTRD`, `xOPGTR`/`xUPGTR`, and
left/right `xOPMTR`/`xUPMTR` application across all four scalar types.

Generalized-reduction examples cover `xSPGST`/`xHPGST` for all four scalar
types. They factor positive-definite `B`, reduce `A`, solve the standard packed
problem, compare with the high-level generalized drivers, back-transform
eigenvectors for all three problem forms, and check generalized residuals.

BLAS rank-update examples cover `SSPR`, `DSPR`, `SSPR2`, `DSPR2`, `CHPR`,
`ZHPR`, `CHPR2`, and `ZHPR2`. They compare every logical entry with an explicit
update formula and check Hermitian conjugation and real diagonals.

Symmetric/Hermitian indefinite examples cover factorization, solve, inverse,
condition estimation, and iterative refinement for every applicable scalar.
They also cover `LANSP` for `f32`, while the test suite covers every public norm
scalar and structure combination.

Packed eigensolver examples cover:

- basic `SPEV`/`HPEV` drivers for all valid scalar families;
- divide-and-conquer `SPEVD`/`HPEVD` drivers for all valid scalar families;
- selected `SPEVX`/`HPEVX` drivers by index and value range;
- generalized `SPGV`/`HPGV` basic drivers for both precisions;
- generalized `SPGVD`/`HPGVD` and `SPGVX`/`HPGVX` drivers in representative
  `f64` and `Complex64` examples.

The eigensolver examples check ordered eigenvalues, residuals, normalization,
and orthogonal/unitary orthogonality without relying on eigenvector signs or
phases. Generalized type-1 examples validate `A v ≈ lambda B v`.

The smaller non-`lapack_` examples demonstrate the main matrix APIs, owned and
view-backed storage, and destructive mutable-view factorization. Numerical API
tests supplement the examples with edge sizes, validation errors, allocation
policy, multiple right-hand sides, singular matrices, and upper/lower logical
access.
