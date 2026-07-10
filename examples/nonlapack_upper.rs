//! Non-LAPACK operations for `PackedUpper`, including complex scalars.
//! Run with: `cargo run --example nonlapack_upper`

use matrixpacked::{PackedUpper, PackedUpperView, PackedUpperViewMut, PackedMatrixError};
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 { Complex64::new(re, im) }

fn main() -> Result<(), PackedMatrixError> {
    // [ 1+i  2    4-i ]
    // [ 0    3+i  5   ]
    // [ 0    0    6   ]
    let mut a = PackedUpper::<Complex64>::from_vec(
        3,
        vec![c(1.0, 1.0), c(2.0, 0.0), c(3.0, 1.0), c(4.0, -1.0), c(5.0, 0.0), c(6.0, 0.0)],
    )?;

    assert_eq!(a.get(0, 2)?, c(4.0, -1.0));
    assert_eq!(a.get(2, 0)?, c(0.0, 0.0));
    assert_eq!(a.packed_index(1, 2), Some(4));
    assert_eq!(a.packed_index(2, 1), None);

    let view: PackedUpperView<'_, Complex64> = a.as_view();
    assert_eq!(view.as_slice().as_ptr(), a.as_slice().as_ptr());

    {
        let mut view_mut: PackedUpperViewMut<'_, Complex64> = a.as_view_mut();
        view_mut.set(0, 1, c(7.0, 2.0))?;
        view_mut[(1, 2)] = c(8.0, -3.0);
    }
    assert_eq!(a.get(0, 1)?, c(7.0, 2.0));
    assert_eq!(a.get(1, 2)?, c(8.0, -3.0));

    let b = PackedUpper::<Complex64>::identity(3)?;
    let sum = &a + &b;
    assert_eq!(sum.get(0, 0)?, c(2.0, 1.0));

    let difference = &sum - &b;
    assert_eq!(difference.as_slice(), a.as_slice());

    let alpha = c(0.0, 2.0);
    let scaled = &a * alpha;
    assert_eq!(scaled.get(0, 0)?, c(-2.0, 2.0));
    let restored = &scaled / alpha;
    assert_eq!(restored.as_slice(), a.as_slice());

    let negated = -&a;
    assert_eq!(negated.get(0, 1)?, c(-7.0, -2.0));

    let squares = a.component_mul(&a)?;
    assert_eq!(squares.get(0, 0)?, c(0.0, 2.0));

    let norm_squared = a.stored_norm_squared();
    let expected: f64 = a.as_slice().iter().map(|z| z.norm_sqr()).sum();
    assert_eq!(norm_squared, expected);

    let mut filled = PackedUpper::<Complex64>::zeros(2)?;
    filled.fill_stored(c(2.0, -1.0));
    assert_eq!(filled.as_slice(), &[c(2.0, -1.0); 3]);

    println!("Display:\n{a}");
    println!("Debug: {a:?}");
    Ok(())
}
