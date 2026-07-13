Work on:

https://github.com/evnekdev/matrixpacked

Start only after the nalgebra test-infrastructure PR has been merged.

Create:

```text
agent/test-packed-storage-semantics
```

Commit and PR title:

```text
Test packed storage semantics
```

# Goal

Add exhaustive tests verifying the logical contents and mutation behavior of all packed matrix storage types against independently constructed nalgebra full matrices.

Use the shared oracle utilities already merged into `tests/oracle/`.

Do not add new numerical functionality in this PR.

# Matrix families

Test:

```text
PackedLower
PackedUpper
PackedSymmetric
PackedSPD
PackedHermitian
```

For every applicable type, test:

```text
Owned
View
ViewMut
```

For scalars:

```text
f32
f64
Complex32
Complex64
```

Respect semantic restrictions:

* `PackedSymmetric<Complex<_>>` is complex symmetric, not Hermitian;
* `PackedHermitian` is complex Hermitian;
* `PackedSPD<Complex<_>>` is HPD.

# Dimensions

Use at least:

```text
n = 0, 1, 2, 3, 4, 5, 8
```

Odd and even dimensions are important.

# Required test groups

## 1. Packed-length formula

Verify:

```text
len = n*(n+1)/2
```

for safe dimensions.

Test overflow handling for dimensions near `usize` limits without attempting huge allocations.

## 2. Exact packed order

For each dimension and triangle, construct a full matrix whose entries encode coordinates, for example:

```text
real part = 100*i + j
imaginary part = 10*i - j
```

Pack it independently using test helpers.

Verify exact storage order from:

```rust
as_slice()
```

This test must not use random values.

## 3. Logical indexing

For every valid `(i, j)`:

* compare packed logical access to nalgebra full matrix access;
* verify symmetric mirroring;
* verify complex-symmetric mirroring does not conjugate;
* verify Hermitian mirroring does conjugate;
* verify lower and upper unstored entries behave correctly;
* verify triangular unstored entries are logical zero if that is the crate contract.

Test invalid indices according to the public API contract.

## 4. Mutable indexing/setters

Using `ViewMut` and owned matrices:

* modify stored entries;
* modify logical mirrored entries if the API permits it;
* verify the underlying packed slot changes correctly;
* verify Hermitian conjugation behavior;
* verify Hermitian diagonal rules;
* compare the resulting full logical matrix with nalgebra.

For Hermitian diagonal writes with nonzero imaginary parts, test the documented behavior:

* rejection;
* normalization;
* or invariant-preserving storage.

Do not guess; inspect the current implementation.

## 5. Owned/view consistency

Create one backing packed buffer and wrap it as:

```text
owned
immutable view
mutable view
```

Verify all report identical dimensions, lengths, and logical entries.

Verify modifications through `ViewMut` are visible in the original backing slice.

## 6. Clone and conversion semantics

Where implemented, test:

* view to owned;
* mutable view to owned;
* structural conversion methods;
* SPD/HPD to symmetric/Hermitian conversions;
* preserving exact packed storage where expected.

## 7. Iteration

Test every iterator exposed by the crate:

* storage iteration;
* logical row/column iteration;
* diagonal iteration;
* any coordinate/value iterators.

Compare yielded data against nalgebra.

Check exact order where order is part of the API.

## 8. Debug and Display

For small matrices, verify formatting corresponds to the logical full matrix, especially:

* triangular zero regions;
* symmetric mirrored entries;
* Hermitian conjugation;
* complex formatting.

Avoid excessively brittle whitespace tests unless formatting is explicitly stable.

Prefer comparing normalized lines or key matrix content.

# Property-based tests

Use `proptest` for dimensions in a modest range, such as:

```text
0..12
```

and random finite values.

Properties:

* pack then logical-expand equals the source structured matrix;
* every logical access agrees with nalgebra;
* View and Owned produce the same logical matrix;
* mutation of one stored slot changes exactly the intended logical entries;
* Hermitian result remains Hermitian;
* symmetric result remains symmetric.

Reject NaN and infinity in generated data unless testing those explicitly.

# Validation

Run the normal validation suite plus:

```bash
cargo test --test nalgebra_oracle storage
```

Adapt the command to the actual integration-test name.

The PR description must list every matrix/storage/scalar combination tested.

Open a draft PR and finish only with:

**Safe to rebase and merge.**
