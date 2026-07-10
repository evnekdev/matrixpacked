use matrixpacked::{PackedSPD, PackedSPDView, PackedSPDViewMut};

fn main() {
    // SPD matrix [4 1 1; 1 3 0; 1 0 2], lower-packed by columns.
    let mut matrix = PackedSPD::<f64>::from_vec(3, vec![4.0, 1.0, 1.0, 3.0, 0.0, 2.0]).unwrap();
    assert_eq!(matrix.get(0, 2).unwrap(), 1.0);
    matrix.set(2, 0, 2.0).unwrap();
    let view: PackedSPDView<'_, f64> = matrix.as_view();
    assert_eq!(view.get(0, 2).unwrap(), 2.0);
    let mut view_mut: PackedSPDViewMut<'_, f64> = matrix.as_view_mut();
    view_mut.set(1, 1, 4.0).unwrap();

    // nalgebra-like logical matrix formatting.
    println!("Display:\n{matrix}");
    println!("Debug: {matrix:?}");
}
