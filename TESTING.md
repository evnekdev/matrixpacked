# Testing matrixpacked

The test system has three tiers. All generated floating-point values in the
default property strategies are finite and bounded. The crate does not define
special numerical-driver semantics for NaN or infinity, so non-finite inputs
are intentionally outside the oracle properties rather than being allowed to
pass comparisons vacuously.

## Tier 1: fast tests

Ordinary tests include deterministic examples and a small property sample of
16 cases per property with a fixed runner seed. Run them whenever changing the
crate:

```bash
cargo test --all-targets --features openblas-static
```

On Windows, `openblas-src` does not support its non-vcpkg static build. Use MKL:

```powershell
cargo test --all-targets --features intel-mkl-static
```

The fast suite is intended to take seconds after dependencies are built.

## Tier 2: property tests

Run only tests whose names contain `property` and raise the controlled case
count when desired:

```bash
MATRIXPACKED_PROPTEST_CASES=96 cargo test --test nalgebra_oracle property --features openblas-static
```

PowerShell equivalent:

```powershell
$env:MATRIXPACKED_PROPTEST_CASES = "96"
cargo test --test nalgebra_oracle property --features intel-mkl-static
```

The property suite covers storage and views, arithmetic and rank updates,
single- and multi-right-hand-side solves, left/right inverse residuals,
iterative refinement, ordinary eigensolvers, and generalized eigensolvers.
CI runs 96 fixed-seed cases on pull requests. Runtime is normally under a
minute once the native backend is available.

Proptest shrinks a failing input and reports the minimized arguments. Oracle
assertions additionally report the scalar, matrix family, dimension, seed,
packed storage, dense matrix, operation options, and numerical residual or
difference. To replay with a particular runner seed and case count:

```bash
MATRIXPACKED_PROPTEST_SEED=557075679 MATRIXPACKED_PROPTEST_CASES=96 \
  cargo test --test nalgebra_oracle property --features openblas-static -- --nocapture
```

Proptest's persisted regression entries, when generated, are also replayed
automatically. Generator payload seeds are printed as part of minimized cases.

## Tier 3: extended numerical stress

Extended tests are inert during ordinary `cargo test`. Enable them explicitly:

```bash
MATRIXPACKED_EXTENDED_TESTS=1 \
  cargo test --test nalgebra_oracle extended_property --features openblas-static -- --nocapture
```

They exercise dimensions through 48, multiple fixed seeds, multi-RHS solves,
cross-algorithm eigensolver agreement, and deliberately ill-conditioned
matrices. Expect roughly one to several minutes after compilation, depending
on backend and host. GitHub Actions runs this tier weekly and on manual
dispatch using OpenBLAS. Windows MKL receives a smaller deterministic backend
smoke suite on every push and pull request.

## Conditioning categories

Generators distinguish these expectations:

- `well_conditioned_spd_f64`: normal solve, inverse, and refinement tolerances;
- `moderately_conditioned_spd_f64`: broader eigenvalue scales;
- `deliberately_ill_conditioned_spd_f64`: relaxed residual-only stress checks;
- `singular_psd_f64`: failure or near-zero-spectrum tests, never successful-solve assumptions.

## Oracle limitations

The dense nalgebra oracle is independent of packed indexing and LAPACK driver
storage, but it uses the same host floating-point arithmetic. Tolerances are
scale-aware and differ between 32- and 64-bit scalars. Eigenvector sign/phase
is not unique, and repeated eigenvalues are compared by invariant subspaces.
Very ill-conditioned problems are checked with normalized residuals rather
than elementwise solution equality. Backend-specific convergence failures are
reported rather than retried, preserving deterministic flakiness behavior.
