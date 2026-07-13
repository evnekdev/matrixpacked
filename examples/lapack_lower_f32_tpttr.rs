use matrixpacked::PackedLower;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let packed = PackedLower::from_vec(3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let full = packed.to_full_triangular()?;
    assert_eq!(
        full.as_slice(),
        &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]
    );
    assert_eq!(
        PackedLower::from_full_triangular(&full)?.as_slice(),
        packed.as_slice()
    );
    Ok(())
}
