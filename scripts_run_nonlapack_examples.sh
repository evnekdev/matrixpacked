#!/usr/bin/env sh
set -eu

for example in \
  nonlapack_lower \
  nonlapack_upper \
  nonlapack_symmetric \
  nonlapack_spd \
  nonlapack_hermitian
do
  echo "==> cargo run --example $example"
  cargo run --example "$example"
done
