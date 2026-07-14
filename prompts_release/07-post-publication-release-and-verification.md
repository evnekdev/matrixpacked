# Prompt 07 — Complete GitHub release and verify publication

Work on https://github.com/evnekdev/matrixpacked

Start only after `matrixpacked 0.1.0` is confirmed visible on crates.io.

## Goal

Connect the registry release to the exact Git commit, verify docs.rs, add live badges/links, and establish the post-release baseline.

Do not change numerical code.

## Verify crates.io

Confirm:

- crate/version;
- publish timestamp;
- owners;
- license;
- README;
- repository/documentation links;
- dependencies;
- features;
- categories/keywords;
- downloadable source contents.

Record discrepancies.

## Verify docs.rs

Confirm version `0.1.0` docs build successfully and include `nalgebra-interop` APIs without a native provider.

Inspect landing page, source links, public items, and intra-doc links.

If docs.rs fails, inspect logs and prepare a `0.1.1` fix. Never try to replace `0.1.0`.

## Tag the exact published commit

Identify the exact commit used for `cargo publish`.

Create and push an annotated tag:

```bash
git tag -a v0.1.0 <published-commit-sha> -m "matrixpacked 0.1.0"
git show v0.1.0
git push origin v0.1.0
```

The tag must not point to a later unrelated commit. Do not casually move a published release tag.

## GitHub release

Create a normal GitHub release from `v0.1.0`:

```text
matrixpacked 0.1.0
```

Use the reviewed `RELEASE_NOTES_0.1.0.md` body. Do not upload the `.crate` archive manually.

## Post-release documentation PR

If changes are needed, create `agent/post-release-0.1.0` and open **Document matrixpacked 0.1.0 release**.

Allowed changes:

- live crates.io badge;
- live docs.rs badge;
- changelog comparison links;
- release checklist publication status;
- crates.io/docs.rs links;
- post-release status note.

Do not bump to `0.1.1` without actual patch development.

Add changelog links:

```markdown
[Unreleased]: https://github.com/evnekdev/matrixpacked/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/evnekdev/matrixpacked/releases/tag/v0.1.0
```

## Finalize checklist

Record publication, version, published commit, tag, GitHub release, docs.rs status, issue #41, defects, and `0.1.1` recommendation.

## Registry consumer smoke tests

Create temporary external projects using the registry dependency, not a path.

### Core

```toml
matrixpacked = "0.1.0"
```

Compile storage-only code.

### Nalgebra

```toml
matrixpacked = { version = "0.1.0", features = ["nalgebra-interop"] }
```

Compile conversion code.

### Numerical

Enable the appropriate provider and run one solve.

## Issue review

Keep issue #41 open unless separately fixed. Add labels/milestone only if the repository uses them. Create follow-up issues only for concrete defects.

## Documentation PR validation

```bash
cargo fmt --all -- --check
cargo check --all-targets --no-default-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo package --list
git diff --check
```

## Commit

```text
Document matrixpacked 0.1.0 release
```

After any post-release PR is complete, finish with:

**Safe to rebase and merge.**

Also report crates.io status, docs.rs status, tag/release, smoke tests, and immediate follow-ups.
