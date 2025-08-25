#!/bin/bash

wget --no-check-certificate https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz
tar -xf OpenBLAS-0.3.29.tar.gz
cd OpenBLAS-0.3.29/

export CC=gcc
export CXX=g++

make ONLY_CBLAS=1 TARGET=ARMV8 ARCH=arm64 -j32