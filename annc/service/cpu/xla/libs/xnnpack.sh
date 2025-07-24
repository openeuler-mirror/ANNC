# Copyright 2025 Huawei. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/bin/bash

git clone https://github.com/google/XNNPACK
cd XNNPACK
# just to lock the commit and avoid unwanted surprises
git checkout 92ed83bd07c3ba31366010db3ff8e132d8872416
mkdir build
cd build

CFLAGS="-fPIC" cmake .. -DXNNPACK_BUILD_BENCHMARKS=OFF                         \
                        -DXNNPACK_BUILD_TESTS=OFF                              \
                        -DXNNPACK_LIBRARY_TYPE=shared                          \
                        -DCMAKE_BUILD_TYPE=Release
make -j32
