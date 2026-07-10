use matrixpacked::{PackedSymmetric, PackedSymmetricView, PackedSymmetricViewMut};

fn main() {
    // [1 2 3; 2 4 5; 3 5 6], lower-packed by columns.
    let mut matrix = PackedSymmetric::<f64>::from_vec(3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(matrix.get(0, 2).unwrap(), 3.0);
    assert_eq!(matrix.get(2, 0).unwrap(), 3.0);
    matrix.set(0, 2, 10.0).unwrap();
    assert_eq!(matrix.get(2, 0).unwrap(), 10.0);
    let view: PackedSymmetricView<'_, f64> = matrix.as_view();
    assert_eq!(view[(0, 2)], 10.0);
    let mut view_mut: PackedSymmetricViewMut<'_, f64> = matrix.as_view_mut();
    view_mut[(1, 2)] = 20.0;

    // nalgebra-like logical matrix formatting.
    println!("Display:\n{matrix}");
    println!("Debug: {matrix:?}");


    let a = PackedSymmetric::<f64>::from_vec(3, vec![4.0, 1.0, 1.0, 3.0, 0.0, 2.0]).unwrap();
    let y = a.mul_vector(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(y, vec![9.0, 7.0, 7.0]);
    let solved = a.solve_vector(&y).unwrap();
    assert!(solved.iter().zip([1.0, 2.0, 3.0]).all(|(a,b)| (a-b).abs() < 1e-12));
}
