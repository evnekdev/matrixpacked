//! Demonstrates ZPPSVX with computed equilibration.
use matrixpacked::{EquilibrationMode, ExpertSolveOptions, PackedSPD};
use num_complex::Complex64;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Complex64::new;
    let a = PackedSPD::from_vec(2, vec![c(1e-8, 0.), c(0., 0.), c(1e8, 0.)])?;
    let r = a.expert_solve_with_options(
        &[c(1e-8, 0.), c(1e8, 0.)],
        1,
        ExpertSolveOptions {
            equilibration: EquilibrationMode::Compute,
        },
    )?;
    assert!(r.equilibration.is_some());
    assert!(r.backward_error[0] >= 0.);
    assert!((r.solution[0] - c(1., 0.)).norm() < 1e-10);
    Ok(())
}
