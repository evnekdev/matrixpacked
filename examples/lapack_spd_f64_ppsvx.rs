//! Demonstrates DPPSVX with computed equilibration for an ill-scaled matrix.
use matrixpacked::{EquilibrationMode, ExpertSolveOptions, PackedSPD};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = PackedSPD::from_vec(2, vec![1e-8f64, 0., 1e8])?;
    let r = a.expert_solve_with_options(
        &[1e-8, 1e8],
        1,
        ExpertSolveOptions {
            equilibration: EquilibrationMode::Compute,
        },
    )?;
    assert_eq!(r.solution.len(), 2);
    assert!(r.equilibration.is_some());
    assert!(r.reciprocal_condition_number > 0.);
    assert!((r.solution[0] - 1.).abs() < 1e-10 && (r.solution[1] - 1.).abs() < 1e-10);
    Ok(())
}
