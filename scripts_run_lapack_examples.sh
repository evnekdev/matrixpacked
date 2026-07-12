#!/usr/bin/env sh
set -eu

feature_args=""
if [ "${1:-}" = "--openblas-static" ]; then
    feature_args="--features openblas-static"
elif [ "$#" -gt 0 ]; then
    echo "usage: $0 [--openblas-static]" >&2
    exit 2
fi

found=0
for file in examples/lapack_*.rs; do
    [ -f "$file" ] || continue
    found=1
    example=$(basename "$file" .rs)
    echo "==> cargo run --example $example ${feature_args}"
    # Intentional word splitting: feature_args contains zero or two arguments.
    # shellcheck disable=SC2086
    cargo run --quiet --example "$example" $feature_args
done

if [ "$found" -eq 0 ]; then
    echo "No LAPACK examples found in examples/lapack_*.rs" >&2
    exit 1
fi
