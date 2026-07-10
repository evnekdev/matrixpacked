# Non-LAPACK example coverage

The existing `lapack_*` examples remain unchanged. The following additional examples cover operations implemented directly on packed storage without calling BLAS/LAPACK.

| Example | Matrix family | Main coverage |
|---|---|---|
| `nonlapack_lower` | `PackedLower<f64>` | constructors, packed indexing, structural zeros, owned/view/view-mut, `Index`, `IndexMut`, `set`, add/subtract, negation, scalar multiplication/division, assignment operators, componentwise arithmetic, stored norm, zero/identity, formatting, `into_vec` |
| `nonlapack_upper` | `PackedUpper<Complex64>` | complex construction/access, structural zeros, views, mutation, add/subtract, negation, complex scalar multiplication/division, componentwise multiplication, stored norm, filling, formatting |
| `nonlapack_symmetric` | `PackedSymmetric<f64>` | mirrored logical access, shared packed indices, upper/lower mutation, views, add/subtract, scalar operations, assignment operators, componentwise arithmetic, formatting |
| `nonlapack_spd` | `PackedSPD<Complex64>` and `PackedSPD<f64>` | conjugating mirrored access, owned/view/view-mut, structure-preserving addition and addition assignment, zero/identity construction, explanation of deliberately unsupported structure-destroying operators |
| `nonlapack_hermitian` | `PackedHermitian<Complex64>` | conjugating reads/writes, views, add/subtract, negation, fill, zero/identity construction, explanation of deliberately unsupported arbitrary complex scaling |

Run all non-LAPACK examples:

```bash
./scripts_run_nonlapack_examples.sh
```

These examples need no BLAS/LAPACK provider feature because they do not invoke native numerical routines.
