Work on:

https://github.com/evnekdev/matrixpacked

Start only after the deterministic nalgebra-oracle modules are merged.

Create:

```text
agent/property-tests-and-ci
```

Commit and PR title:

```text
Add property tests and CI coverage
```

# Goal

Turn the deterministic nalgebra oracle suite into a broad, repeatable property-testing and CI system without making normal development unreasonably slow.

# Test tiers

Create three tiers:

## Tier 1: Fast deterministic tests

Run on every:

```text
cargo test
```

Characteristics:

* fixed seeds;
* small dimensions;
* limited case count;
* completes quickly.

## Tier 2: Property tests

Run in CI and locally through:

```bash
cargo test --test nalgebra_oracle property
```

or a dedicated feature/environment variable.

Use `proptest` with controlled case counts.

## Tier 3: Extended numerical stress tests

Run manually or in scheduled CI.

Suggested trigger:

```text
MATRIXPACKED_EXTENDED_TESTS=1
```

Characteristics:

* larger dimensions;
* more seeds;
* ill-conditioned matrices;
* multiple backends;
* slower runtime.

Do not run extended tests by default on every local `cargo test`.

# Properties to cover

## Storage

* logical expansion matches independent full matrix;
* packing round trips;
* view/owned equivalence;
* mutation preserves structure.

## Arithmetic

* packed multiplication matches full multiplication;
* scalar operations match full operations;
* rank updates match full updates.

## Solves

* generated exact solution is recovered;
* normalized residual is small;
* multi-RHS matches column-by-column solves.

## Inverses

* left and right identity residuals are small.

## Refinement

* residual does not worsen beyond a tiny tolerance;
* usually improves perturbed solutions.

## Eigensolvers

* residuals;
* orthogonality/unitarity;
* reconstruction;
* cross-algorithm eigenvalue agreement.

## Generalized eigensolvers

* generalized residuals;
* normalization;
* transformed-standard agreement.

# Shrinking and diagnostics

Proptest failures must produce reproducible minimized cases.

When a property fails, report:

* random seed or proptest case;
* scalar type;
* matrix family;
* dimension;
* packed storage;
* full matrix;
* operation options;
* residual or difference.

Avoid opaque generic panic messages.

# NaN and infinity

Default generators should use finite bounded values.

Create separate explicit tests for NaN/infinity behavior if the crate defines it.

Do not allow accidental NaNs to make comparisons vacuously pass.

# Conditioning controls

Separate generators into:

```text
well-conditioned
moderately conditioned
deliberately ill-conditioned
singular
```

Do not expect the same tolerance or success behavior from all categories.

# CI workflow

Add or update GitHub Actions to test:

* Linux with the primary BLAS/LAPACK backend;
* Windows if the repository supports a reliable backend;
* stable Rust;
* all ordinary tests;
* nalgebra oracle tests;
* formatting;
* clippy if already part of project policy.

Do not add a matrix so large that CI becomes impractical.

Recommended initial CI split:

```text
fast tests: every push/PR
property tests: every PR
extended stress tests: scheduled weekly or manual dispatch
```

# Backend matrix

Where both OpenBLAS and MKL are supported, run the full numerical oracle on one backend and a smaller smoke suite on the other.

The purpose is to detect backend integration differences without doubling all CI cost.

# Flakiness policy

No test should depend on nondeterministic thread scheduling or OS random sources.

All random tests must be deterministic or print reproducible seeds.

If a test requires a probabilistic tolerance, redesign it rather than retrying automatically.

# Documentation

Add a testing section to the README or a dedicated:

```text
TESTING.md
```

Document:

* test tiers;
* commands;
* backend requirements;
* extended test environment variable;
* how to reproduce a property failure;
* expected runtime;
* oracle limitations.

# Validation

Run the full local suite available in the environment.

Inspect GitHub Actions YAML carefully.

Open a draft PR and finish only with:

**Safe to rebase and merge.**
