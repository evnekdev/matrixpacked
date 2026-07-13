# LAPACK example coverage

This directory contains **124** individual checked examples.

Run one example with:

```bash
cargo run --example lapack_lower_f64_tpmv --features openblas-static
```

Run all examples on a Unix-like shell with:

```bash
for example in $(cargo metadata --no-deps --format-version 1 | jq -r ".packages[0].targets[] | select(.kind[] == \"example\") | .name" | grep "^lapack_"); do
  cargo run --quiet --example "$example" --features openblas-static || exit 1
done
```

## Generated examples

The triangular packed matrix set covers every upper/lower and
`f32`/`f64`/`Complex32`/`Complex64` combination for `TPMV`, `TPSV`, `TPTRS`,
`TPTRI`, `TPCON`, `TPRFS`, and `LANTP`.

Packed iterative-refinement coverage also includes `PPRFS` for all four scalar
families, real `SPRFS`, and complex `HPRFS`. Each example starts from a
perturbed solution and checks both the refined solution and the per-right-hand-side
forward/backward error arrays.

Inverse coverage includes lower and upper `TPTRI` for all four scalar families,
`PPTRI` and complex-symmetric/real-symmetric `SPTRI` for all four scalar
families, and complex `HPTRI`. Every inverse example checks numerical packed
entries; the API test suite additionally verifies matrix-vector round trips,
owned allocation, mutable views, diagonal modes, singular errors, and edge sizes.

- `lapack_hermitian_c32_hpmv`
- `lapack_hermitian_c32_hpcon`
- `lapack_hermitian_c32_hptrf`
- `lapack_hermitian_c32_hptri`
- `lapack_hermitian_c32_hptrs`
- `lapack_hermitian_c32_hprfs`
- `lapack_hermitian_c64_hpmv`
- `lapack_hermitian_c64_hpcon`
- `lapack_hermitian_c64_hptrf`
- `lapack_hermitian_c64_hptri`
- `lapack_hermitian_c64_hptrs`
- `lapack_hermitian_c64_hprfs`
- `lapack_lower_c32_tpmv`
- `lapack_lower_c32_tptri`
- `lapack_lower_c32_tptrs`
- `lapack_lower_c64_tpmv`
- `lapack_lower_c64_tptri`
- `lapack_lower_c64_tptrs`
- `lapack_lower_f32_tpmv`
- `lapack_lower_f32_tptri`
- `lapack_lower_f32_tptrs`
- `lapack_lower_f64_tpmv`
- `lapack_lower_f64_tptri`
- `lapack_lower_f64_tptrs`
- `lapack_spd_c32_pmv`
- `lapack_spd_c32_ppcon`
- `lapack_spd_c32_pptrf`
- `lapack_spd_c32_pptri`
- `lapack_spd_c32_pptrs`
- `lapack_spd_c32_pprfs`
- `lapack_spd_c64_pmv`
- `lapack_spd_c64_ppcon`
- `lapack_spd_c64_pptrf`
- `lapack_spd_c64_pptri`
- `lapack_spd_c64_pptrs`
- `lapack_spd_c64_pprfs`
- `lapack_spd_f32_pmv`
- `lapack_spd_f32_ppcon`
- `lapack_spd_f32_pptrf`
- `lapack_spd_f32_pptri`
- `lapack_spd_f32_pptrs`
- `lapack_spd_f32_pprfs`
- `lapack_spd_f64_pmv`
- `lapack_spd_f64_ppcon`
- `lapack_spd_f64_pptrf`
- `lapack_spd_f64_pptri`
- `lapack_spd_f64_pptrs`
- `lapack_spd_f64_pprfs`
- `lapack_symmetric_c32_sptrf`
- `lapack_symmetric_c32_spcon`
- `lapack_symmetric_c32_sptri`
- `lapack_symmetric_c32_sptrs`
- `lapack_symmetric_c64_sptrf`
- `lapack_symmetric_c64_spcon`
- `lapack_symmetric_c64_sptri`
- `lapack_symmetric_c64_sptrs`
- `lapack_symmetric_f32_spmv`
- `lapack_symmetric_f32_spcon`
- `lapack_symmetric_f32_sptrf`
- `lapack_symmetric_f32_sptri`
- `lapack_symmetric_f32_sptrs`
- `lapack_symmetric_f32_sprfs`
- `lapack_symmetric_f64_spmv`
- `lapack_symmetric_f64_spcon`
- `lapack_symmetric_f64_sptrf`
- `lapack_symmetric_f64_sptri`
- `lapack_symmetric_f64_sptrs`
- `lapack_symmetric_f64_sprfs`
- `lapack_upper_c32_tpmv`
- `lapack_upper_c32_tptri`
- `lapack_upper_c32_tptrs`
- `lapack_upper_c64_tpmv`
- `lapack_upper_c64_tptri`
- `lapack_upper_c64_tptrs`
- `lapack_upper_f32_tpmv`
- `lapack_upper_f32_tptri`
- `lapack_upper_f32_tptrs`
- `lapack_upper_f64_tpmv`
- `lapack_upper_f64_tptri`
- `lapack_upper_f64_tptrs`
