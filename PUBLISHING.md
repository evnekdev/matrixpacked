# Publishing `matrixpacked` 0.1.0

This runbook is for the first crates.io publication. Publishing a version is
irreversible: a published version cannot be overwritten, and yanking does not
delete it. A correction requires publishing `0.1.1`.

1. Fetch the remote state:

   ```text
   git fetch --all --prune
   ```

2. Switch to `master` and fast-forward only:

   ```text
   git switch master
   git pull --ff-only
   ```

3. Verify that `git status --short` has no output. Do not use
   `--allow-dirty`, `--no-verify`, or `--no-metadata`.

4. Verify the exact merged commit approved for the release and confirm its
   required GitHub Actions CI is green. Confirm `Cargo.toml` still declares
   `matrixpacked` version `0.1.0`, no `v0.1.0` tag exists, and crates.io does
   not already show that version.

5. Run the non-uploading validation again:

   ```text
   cargo publish --dry-run
   ```

6. Configure crates.io authentication only in the local Cargo environment.
   Never put a token in chat, source control, shell history, logs, a workflow,
   or a command shown to another person.

7. Obtain explicit approval that names the exact commit, for example:

   ```text
   Publish matrixpacked 0.1.0 from commit <exact-sha> to crates.io now.
   ```

8. Only after that approval, run:

   ```text
   cargo publish
   ```

9. If Cargo times out while polling, first check crates.io for
   `matrixpacked 0.1.0`: the upload may already have succeeded. Never retry
   immediately without checking the registry state.

10. Verify the publication is visible on crates.io before creating `v0.1.0`, a
    GitHub release, or any release tag.

Do not create publishing automation or a GitHub Actions secret workflow for
this first release unless explicitly authorized later.
