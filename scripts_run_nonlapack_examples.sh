#!/usr/bin/env sh
set -eu

found=0
for file in examples/*.rs; do
    [ -f "$file" ] || continue

    example=$(basename "$file" .rs)
    case "$example" in
        lapack_*)
            continue
            ;;
    esac

    found=1
    case "$example" in
        nalgebra_*)
            echo "==> cargo run --example $example --features nalgebra-interop"
            cargo run --quiet --example "$example" --features nalgebra-interop
            ;;
        *)
            echo "==> cargo run --example $example"
            cargo run --quiet --example "$example"
            ;;
    esac
done

if [ "$found" -eq 0 ]; then
    echo "No non-LAPACK examples found in examples/*.rs" >&2
    exit 1
fi
