#!/bin/sh
set -eu

feature_args=""
if [ "${1:-}" = "--openblas-static" ]; then
    feature_args="--features openblas-static"
fi

for file in examples/lapack_*.rs; do
    example=$(basename "$file" .rs)
    echo "==> $example"
    # shellcheck disable=SC2086
    cargo run --quiet --example "$example" $feature_args
done
