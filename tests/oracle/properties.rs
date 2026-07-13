use proptest::prelude::*;

pub fn matrix_dimension() -> impl Strategy<Value = usize> {
    0usize..=24
}
pub fn nonsingular_dimension() -> impl Strategy<Value = usize> {
    1usize..=24
}
pub fn indefinite_dimension() -> impl Strategy<Value = usize> {
    2usize..=24
}
pub fn deterministic_seed() -> impl Strategy<Value = u64> {
    any::<u64>()
}
pub fn rhs_count() -> impl Strategy<Value = usize> {
    1usize..=8
}
