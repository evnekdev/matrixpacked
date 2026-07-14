# Prompt 01 — Audit and eliminate warnings

Work on https://github.com/evnekdev/matrixpacked

## Workflow

1. Fetch remote changes.
2. Switch to `master` and pull with fast-forward only.
3. Confirm a clean working tree.
4. Create `agent/eliminate-build-warnings` from current `master`.
5. Do not modify `master` directly.
6. Keep GitHub issue #41 deferred unless a lint command is literally impossible without it.
7. Open a draft PR titled **Eliminate build warnings**.
8. Finish only when no more commits are planned, with:

**Safe to rebase and merge.**

## Context

The crate now includes packed arithmetic, BLAS/LAPACK operations, factorization, condition/refinement APIs, simple and expert drivers, eigensolvers, reductions, format conversions, diagnostics, optional `nalgebra-interop`, extensive nalgebra oracle tests, property/stress tests, and Linux/OpenBLAS plus Windows/MKL CI.

Earlier PR descriptions mentioned historical warnings such as private bounds, dead code, and malformed rustdoc generic notation. Reproduce the current inventory; do not assume those warnings remain.

## Goal

Make all intentionally supported configurations warning-free by fixing causes rather than suppressing diagnostics.

Do not add broad attributes:

```rust
#![allow(warnings)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::all)]
```

A narrow `#[allow(...)]` is acceptable only when unavoidable and documented beside the attribute.

Do not introduce `deny(missing_docs)` here; Prompt 02 owns documentation completeness.

## Inventory before editing

Run and record all current warnings.

### Provider-free paths

```bash
cargo clean
cargo check --lib --no-default-features
cargo check --all-targets --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo check --examples --no-default-features
cargo check --examples --features nalgebra-interop
```

Unresolved native symbols while linking numerical tests/examples are provider limitations, not compiler-warning failures.

### Linked backend paths

On Windows:

```bash
cargo check --all-targets --features intel-mkl-static
cargo test --no-run --features intel-mkl-static
cargo check --all-targets --features "nalgebra-interop,intel-mkl-static"
cargo test --no-run --features "nalgebra-interop,intel-mkl-static"
```

On Linux, use the repository’s documented OpenBLAS setup.

### Clippy

```bash
cargo clippy --lib --no-default-features -- -D warnings
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
```

Also run the provider-linked all-target form appropriate to the host.

### Rustdoc

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --no-default-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
```

## Fixing rules

### Private bounds/interfaces

For public APIs using private traits, choose the correct architectural fix:

- keep backend traits private and move dispatch behind private helpers;
- expose a public semantic trait only if it is genuinely part of the API;
- use concrete supported scalar implementations where clearer.

Do not silence `private_bounds` or expose FFI backend traits merely to satisfy linting.

### Dead code

Delete obsolete private code, integrate accidentally disconnected code, or move test-only helpers under `#[cfg(test)]`. Do not delete public APIs because they are unused internally.

### Feature-dependent imports

Use precise `#[cfg(...)]` placement. Ensure enabling/disabling `nalgebra-interop` creates no unused imports. Preserve intentionally feature-gated provider imports such as `use intel_mkl_src as _;`.

### LAPACK-shaped signatures

Many private backend functions mirror LAPACK and legitimately take many arguments. Prefer one narrowly documented `clippy::too_many_arguments` allowance at a private trait/macro boundary if restructuring would obscure the ABI contract. Do not spread allowances across every implementation.

### Numeric casts

Use existing checked conversion helpers for dimensions, leading dimensions, workspaces, pivots, and RHS counts. Do not replace checked logic with `as i32`.

### Unsafe code

Keep unsafe scopes minimal. For Rust 2024 unsafe-operation warnings, add explicit local `unsafe { ... }` blocks rather than suppressing the lint.

### Rustdoc warnings

Fix broken links, invalid HTML, bare URLs, malformed tables, and generic types such as `Vec<T>` not enclosed in backticks.

### Tests and examples

Every tracked test helper and example—including optional nalgebra examples—must compile warning-free under its required features.

## Scope restrictions

Do not:

- redesign the public conversion API;
- implement issue #41;
- add new numerical functionality;
- perform the comprehensive documentation rewrite;
- overhaul CI.

Create follow-up issues for larger findings.

## Validation

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo check --all-targets --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo clippy --lib --no-default-features -- -D warnings
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
git diff --check
```

Run provider-linked `cargo check --all-targets` and `cargo test --no-run` on the host.

Inspect:

```bash
git diff master...HEAD --stat
git diff master...HEAD
```

## PR description

List the initial warning inventory, root-cause fixes, any narrow retained allowances, tested configurations, untested configurations, and confirmation that issue #41 remains deferred.
