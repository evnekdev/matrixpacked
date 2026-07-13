//! Packed upper-triangular operations backed by BLAS/LAPACK.
//! `mul_vector` uses the `xTPMV` BLAS family to compute a packed triangular
//! matrix-vector product. `solve_vector` uses LAPACK's `xTPTRS` family to solve
//! a triangular system; `x` denotes the scalar-type prefix (`S`, `D`, `C`, or `Z`).

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

    let a = PackedUpper::<f64>::from_vec(3, vec![2.0, 3.0, 1.0, 1.0, 4.0, 5.0]).unwrap();
    let x = a.mul_vector(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(x, vec![11.0, 14.0, 15.0]);
    let solved = a.solve_vector(&x).unwrap();
    assert!(
        solved
            .iter()
            .zip([1.0, 2.0, 3.0])
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
}
