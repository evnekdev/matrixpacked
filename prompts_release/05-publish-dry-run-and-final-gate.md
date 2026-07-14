# Prompt 05 — Run publication dry run and prepare the final gate

Work on https://github.com/evnekdev/matrixpacked

Start only after Prompt 04 is merged.

Create `agent/release-0.1.0-final-gate`. Open a draft PR titled **Complete matrixpacked 0.1.0 release gate**. Finish only with:

**Safe to rebase and merge.**

## Goal

Run Cargo's complete crates.io checks without uploading, freeze the release intent, and create a secure human publication runbook.

Do not run a real `cargo publish` in this prompt.

## Registry/name verification

Confirm again that `matrixpacked` is available to the intended crates.io account or already reserved by it. If unavailable, stop.

## Clean release state

Require:

```bash
git status --short
```

with no output.

Confirm:

- latest `master` basis;
- version `0.1.0`;
- matching changelog/release notes;
- no `v0.1.0` tag;
- no existing `matrixpacked 0.1.0` publication;
- green required CI.

## Dry run

Run:

```bash
cargo publish --dry-run
```

Do not use `--allow-dirty` or `--no-verify`.

Also run:

```bash
cargo package --list
cargo package
```

If authentication is unexpectedly needed, use only the user's local Cargo authentication. Never request or print a token.

## Final checks

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo test --doc --features nalgebra-interop
```

Run all tests/examples with the host provider. Confirm green CI.

## Record release intent

Update `RELEASE_CHECKLIST.md`:

```markdown
## 0.1.0 publication gate

- Reviewed release branch/commit: `<sha>`
- Version: `0.1.0`
- `cargo publish --dry-run`: passed
- Package verification: passed
- CI: passed
- Human publish approval: pending
```

Because the PR merge changes the final `master` SHA, this is the reviewed pre-merge SHA; Prompt 06 must verify the final merged SHA again.

Do not create a tag here.

## Create PUBLISHING.md

Write a secure manual runbook:

1. fetch remote;
2. switch to `master`;
3. fast-forward pull;
4. verify clean tree;
5. verify exact commit and CI;
6. run `cargo publish --dry-run` again;
7. configure crates.io authentication locally;
8. never expose the token;
9. run `cargo publish` only after explicit approval;
10. check crates.io if Cargo times out while polling because upload may have succeeded;
11. never immediately retry without checking registry state;
12. verify publication before tagging.

Explain irreversibility:

- a published version cannot be overwritten;
- yanking does not delete it;
- corrections require `0.1.1`.

## Security restrictions

Do not commit or display tokens, credential files, environment dumps, authentication headers, or shell history.

Do not create publish automation or a GitHub Actions secret workflow for the first release unless explicitly requested later.

## Validation

```bash
cargo publish --dry-run
cargo package
git diff --check
git diff master...HEAD
```

## Commit

```text
Complete matrixpacked 0.1.0 release gate
```

## PR description

Include dry-run result, exact checks, secure publication procedure, known limitations, and confirmation that no upload occurred.
