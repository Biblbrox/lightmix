#!/bin/bash
set -e

export PKG_CONFIG_PATH="/usr/lib/aarch64-linux-gnu/pkgconfig"

# . $HOME/.cargo/env >> ~/.bashrc
export PATH=/home/build/.cargo/bin/:$PATH
export CC=gcc-10
export CXX=g++-10
export PATH="$HOME/zig-aarch64-linux-0.15.2:$PATH"

#wget https://ziglang.org/download/0.15.2/zig-aarch64-linux-0.15.2.tar.xz
#tar xf zig-aarch64-linux-0.15.2.tar.xz
#ls -alh
#echo $PATH

exec "$@"
