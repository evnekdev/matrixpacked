//! Non-LAPACK operations for `PackedSymmetric`.
//! Run with: `cargo run --example nonlapack_symmetric`

use matrixpacked::{PackedSymmetric, PackedSymmetricView, PackedSymmetricViewMut, PackedMatrixError};

fn main() -> Result<(), PackedMatrixError> {
    // Lower-packed storage for:
    // [ 1 2 3 ]
    // [ 2 4 5 ]
    // [ 3 5 6 ]
    let mut a = PackedSymmetric::<f64>::from_vec(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    // Mirrored coordinates refer to the same physical packed element.
    assert_eq!(a.get(0, 2)?, 3.0);
    assert_eq!(a.get(2, 0)?, 3.0);
    assert_eq!(a.packed_index(0, 2), a.packed_index(2, 0));
    assert!(a.get_stored(0, 2).is_none());
    assert_eq!(a.try_get(0, 2)?, &3.0);

    // Setting either triangle updates the same stored value.
    a.set(0, 2, 9.0)?;
    assert_eq!(a.get(2, 0)?, 9.0);

    let view: PackedSymmetricView<'_, f64> = a.as_view();
    assert_eq!(view.as_slice().as_ptr(), a.as_slice().as_ptr());

    {
        let mut view_mut: PackedSymmetricViewMut<'_, f64> = a.as_view_mut();
        view_mut.set(1, 2, 8.0)?;
        assert_eq!(view_mut.get(2, 1)?, 8.0);
    }

    let b = PackedSymmetric::<f64>::from_fn(3, |row, col| (row + col + 1) as f64)?;
    let sum = &a + &b;
    let back = &sum - &b;
    assert_eq!(back.as_slice(), a.as_slice());

    let twice = &a * 2.0;
    let original = &twice / 2.0;
    assert_eq!(original.as_slice(), a.as_slice());

    let negative = -&a;
    assert_eq!(negative.get(0, 2)?, -9.0);

    let product = a.component_mul(&b)?;
    for (p, (&x, &y)) in product.as_slice().iter().zip(a.as_slice().iter().zip(b.as_slice())) {
        assert_eq!(*p, x * y);
    }

    let quotient = product.component_div(&b)?;
    assert_eq!(quotient.as_slice(), a.as_slice());

    let mut assigned = PackedSymmetric::<f64>::zeros(3)?;
    assigned += &a;
    assigned *= 0.5;
    assigned /= 0.5;
    assigned -= &a;
    assert_eq!(assigned.as_slice(), &[0.0; 6]);

    println!("Display:\n{a}");
    println!("Debug: {a:?}");
    Ok(())
}
