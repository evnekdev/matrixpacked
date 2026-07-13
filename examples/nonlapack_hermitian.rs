//! Non-LAPACK operations for `PackedHermitian`.
//! Run with: `cargo run --example nonlapack_hermitian`

use matrixpacked::{
    PackedHermitian, PackedHermitianView, PackedHermitianViewMut, PackedMatrixError,
};
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn main() -> Result<(), PackedMatrixError> {
    // [ 1      2-i   4+2i ]
    // [ 2+i    3     5-i  ]
    // [ 4-2i   5+i   6    ]
    let mut a = PackedHermitian::<Complex64>::from_vec(
        3,
        vec![
            c(1.0, 0.0),
            c(2.0, 1.0),
            c(4.0, -2.0),
            c(3.0, 0.0),
            c(5.0, 1.0),
            c(6.0, 0.0),
        ],
    )?;

    assert_eq!(a.get(2, 0)?, c(4.0, -2.0));
    assert_eq!(a.get(0, 2)?, c(4.0, 2.0));
    assert!(a.get_stored(0, 2).is_none());

    // Setting an upper element writes its conjugate into lower storage.
    a.set(0, 1, c(7.0, -3.0))?;
    assert_eq!(a.get(0, 1)?, c(7.0, -3.0));
    assert_eq!(a.get(1, 0)?, c(7.0, 3.0));

    let view: PackedHermitianView<'_, Complex64> = a.as_view();
    assert_eq!(view.as_slice().as_ptr(), a.as_slice().as_ptr());

    {
        let mut view_mut: PackedHermitianViewMut<'_, Complex64> = a.as_view_mut();
        view_mut.set(1, 2, c(8.0, 4.0))?;
    }
    assert_eq!(a.get(1, 2)?, c(8.0, 4.0));
    assert_eq!(a.get(2, 1)?, c(8.0, -4.0));

    let b = PackedHermitian::<Complex64>::identity(3)?;
    let sum = &a + &b;
    let restored = &sum - &b;
    assert_eq!(restored.as_slice(), a.as_slice());

    let negative = -&a;
    assert_eq!(negative.get(0, 1)?, c(-7.0, 3.0));
    assert_eq!(negative.get(1, 0)?, c(-7.0, -3.0));

    // Arbitrary complex scalar multiplication is intentionally absent because
    // it can destroy Hermitian symmetry. Addition, subtraction and negation do
    // preserve the structure and are therefore implemented.
    let mut filled = PackedHermitian::<Complex64>::zeros(2)?;
    filled.fill_stored(c(2.0, 0.0));
    assert_eq!(filled.as_slice(), &[c(2.0, 0.0); 3]);

    println!("Display:\n{a}");
    println!("Debug: {a:?}");
    Ok(())
}
