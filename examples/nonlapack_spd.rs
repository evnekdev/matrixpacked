//! Non-LAPACK, structure-preserving operations for `PackedSPD`.
//! Run with: `cargo run --example nonlapack_spd`

use matrixpacked::{PackedSPD, PackedSPDView, PackedSPDViewMut, PackedMatrixError};
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 { Complex64::new(re, im) }

fn main() -> Result<(), PackedMatrixError> {
    // Hermitian positive-definite example in lower-packed storage:
    // [ 4      1-i ]
    // [ 1+i    3   ]
    let mut a = PackedSPD::<Complex64>::from_vec(2, vec![c(4.0, 0.0), c(1.0, 1.0), c(3.0, 0.0)])?;

    assert_eq!(a.get(1, 0)?, c(1.0, 1.0));
    assert_eq!(a.get(0, 1)?, c(1.0, -1.0));

    // Writing through the upper triangle stores the conjugate below it.
    a.set(0, 1, c(2.0, -3.0))?;
    assert_eq!(a.get(0, 1)?, c(2.0, -3.0));
    assert_eq!(a.get(1, 0)?, c(2.0, 3.0));
    assert_eq!(a.as_slice()[1], c(2.0, 3.0));

    let view: PackedSPDView<'_, Complex64> = a.as_view();
    assert_eq!(view.as_slice().as_ptr(), a.as_slice().as_ptr());

    {
        let mut view_mut: PackedSPDViewMut<'_, Complex64> = a.as_view_mut();
        view_mut.set(0, 0, c(5.0, 0.0))?;
    }
    assert_eq!(a.get(0, 0)?, c(5.0, 0.0));

    // Addition preserves positive definiteness mathematically for SPD operands.
    let identity = PackedSPD::<Complex64>::identity(2)?;
    let sum = &a + &identity;
    assert_eq!(sum.get(0, 0)?, c(6.0, 0.0));
    assert_eq!(sum.get(1, 1)?, c(4.0, 0.0));

    let mut assigned = a;
    assigned += &identity;
    assert_eq!(assigned.as_slice(), sum.as_slice());

    // Deliberately absent: Sub, Neg, arbitrary scalar Mul/Div and componentwise
    // operations, because they do not generally preserve positive definiteness.
    let zeros = PackedSPD::<f64>::zeros(3)?;
    assert_eq!(zeros.as_slice(), &[0.0; 6]);
    let real_identity = PackedSPD::<f64>::identity(3)?;
    assert_eq!(real_identity.as_slice(), &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);

    println!("Display:\n{sum}");
    println!("Debug: {sum:?}");
    Ok(())
}
