#![allow(clippy::drop_non_drop)]

//! Non-LAPACK operations for `PackedLower`.
//! Run with: `cargo run --example nonlapack_lower`

use matrixpacked::{PackedLower, PackedLowerView, PackedLowerViewMut, PackedMatrixError};

fn main() -> Result<(), PackedMatrixError> {
    // Construct in LAPACK lower-packed column order:
    // [ 1 0 0 ]
    // [ 2 4 0 ]
    // [ 3 5 6 ]
    let mut a = PackedLower::<f64>::from_vec(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    assert_eq!(a.shape(), (3, 3));
    assert_eq!(a.dimension(), 3);
    assert_eq!(PackedLower::<f64>::packed_len(3)?, 6);
    assert_eq!(a.packed_index(2, 1), Some(4));
    assert_eq!(a.packed_index(0, 2), None); // structural zero

    // Logical access returns zero above the diagonal.
    assert_eq!(a.get(2, 1)?, 5.0);
    assert_eq!(a.get(0, 2)?, 0.0);
    assert_eq!(a.get_stored(0, 2), None);
    assert!(a.try_get(0, 2).is_err());

    // Index/IndexMut are intended only for physically stored entries.
    assert_eq!(a[(2, 1)], 5.0);
    a[(2, 1)] = 7.0;
    assert_eq!(a.get(2, 1)?, 7.0);

    // Immutable view borrows the same storage.
    let view: PackedLowerView<'_, f64> = a.as_view();
    assert_eq!(view.as_slice().as_ptr(), a.as_slice().as_ptr());
    assert_eq!(view.get(2, 1)?, 7.0);

    // Mutable view changes the owned matrix without allocating.
    {
        let mut view_mut: PackedLowerViewMut<'_, f64> = a.as_view_mut();
        view_mut.set(1, 0, 8.0)?;
        *view_mut.try_get_mut(2, 0)? = 9.0;
    }
    assert_eq!(a.as_slice(), &[1.0, 8.0, 9.0, 4.0, 7.0, 6.0]);

    // from_fn receives logical stored coordinates in packed column order.
    let b = PackedLower::<f64>::from_fn(3, |row, col| (10 * row + col) as f64)?;
    assert_eq!(b.as_slice(), &[0.0, 10.0, 20.0, 11.0, 21.0, 22.0]);

    let sum = &a + &b;
    let difference = &sum - &b;
    assert_eq!(difference.as_slice(), a.as_slice());

    let negative = -&a;
    assert_eq!(negative.as_slice(), &[-1.0, -8.0, -9.0, -4.0, -7.0, -6.0]);

    let scaled = &a * 2.0;
    let restored = &scaled / 2.0;
    assert_eq!(restored.as_slice(), a.as_slice());

    let component_product = a.component_mul(&b)?;
    assert_eq!(
        component_product.as_slice(),
        &[0.0, 80.0, 180.0, 44.0, 147.0, 132.0]
    );
    let component_quotient = component_product.component_div(&b)?;
    // First element is 0/0 = NaN; all remaining stored values recover `a`.
    assert!(component_quotient.as_slice()[0].is_nan());
    assert_eq!(&component_quotient.as_slice()[1..], &a.as_slice()[1..]);

    let expected_norm_squared: f64 = a.as_slice().iter().map(|x| x * x).sum();
    assert_eq!(a.stored_norm_squared(), expected_norm_squared);

    let mut assigned = a.as_view_mut();
    assigned += &b;
    assigned -= &b;
    assigned *= 3.0;
    assigned /= 3.0;
    drop(assigned);
    assert_eq!(a.as_slice(), &[1.0, 8.0, 9.0, 4.0, 7.0, 6.0]);

    let zeros = PackedLower::<f64>::zeros(3)?;
    assert_eq!(zeros.as_slice(), &[0.0; 6]);
    let identity = PackedLower::<f64>::identity(3)?;
    assert_eq!(identity.as_slice(), &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);

    // Display expands structural zeros; Debug follows nalgebra-style columns.
    println!("Display:\n{a}");
    println!("Debug: {a:?}");

    let raw = a.into_vec();
    assert_eq!(raw, vec![1.0, 8.0, 9.0, 4.0, 7.0, 6.0]);
    Ok(())
}
