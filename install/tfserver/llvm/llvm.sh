#!/bin/bash

# This should point to your annc path!
YOUR_ANNC_PATH=
PATH_OF_PATCHES="$ANNC_PATH/install/llvm/"

cat $PATH_OF_PATCHES/annc1.patch | tail -n +7 | cut -c 2- > tmp.patch && patch -p1 < tmp.patch; rm tmp.patch
cat $PATH_OF_PATCHES/annc2.patch | tail -n +7 | cut -c 2- > tmp.patch && patch -p1 < tmp.patch; rm tmp.patch
cat $PATH_OF_PATCHES/annc3.patch | tail -n +7 | cut -c 2- > tmp.patch && patch -p1 < tmp.patch; rm tmp.patch
cat $PATH_OF_PATCHES/annc4.patch | tail -n +7 | cut -c 2- > tmp.patch && patch -p1 < tmp.patch; rm tmp.patch
cat $PATH_OF_PATCHES/concat.patch | tail -n +7 | cut -c 2- > tmp.patch && patch -p1 < tmp.patch; rm tmp.patch

