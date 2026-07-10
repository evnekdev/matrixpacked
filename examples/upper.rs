use matrixpacked::{PackedUpper, PackedUpperView, PackedUpperViewMut};

fn main() {
    // [1 2 3; 0 4 5; 0 0 6], upper-packed by columns.
    let mut matrix = PackedUpper::<f64>::from_vec(3, vec![1.0, 2.0, 4.0, 3.0, 5.0, 6.0]).unwrap();
    assert_eq!(matrix.get(0, 2).unwrap(), 3.0);
    assert_eq!(matrix.get(2, 0).unwrap(), 0.0);
    matrix.set(1, 2, 10.0).unwrap();
    let view: PackedUpperView<'_, f64> = matrix.as_view();
    assert_eq!(view[(1, 2)], 10.0);
    let mut view_mut: PackedUpperViewMut<'_, f64> = matrix.as_view_mut();
    view_mut[(0, 1)] = 20.0;

    // nalgebra-like logical matrix formatting.
    println!("Display:\n{matrix}");
    println!("Debug: {matrix:?}");
}
