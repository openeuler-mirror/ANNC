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

## Use these variables if bazel or go are not on the PATH.
BAZEL=""
GO=""

ANNC_BASE="$HOME/tools"
if [ ! -d "${ANNC_BASE/ANNC}" ]; then
  echo "Please set ANNC_BASE in this script to point to the base directory of ANNC"
fi

# Adjust as appropriate
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export LD_LIBRARY_PATH=/lib64

PATH=$GO:$BAZEL:$PATH
export PATH

# Check for bazel.
if ! command -v bazel >/dev/null 2>&1; then
  echo "Please add bazel to PATH."
  echo "Exiting"
  exit -1
fi

# Check for go.
if ! command -v go >/dev/null 2>&1; then
  echo "Please add go to PATH."
  echo "Exiting"
  exit -1
fi

ANNC="$ANNC_BASE/ANNC"
XNNPACK_BASE="$ANNC/annc/service/cpu/xla/libs"
XNNPACK_DIR="$XNNPACK_BASE/XNNPACK"

# At the moment XNNPAK is cloned and compiled by a script. The script is only
# run if the directory containing XNNPACK does not exist.
if [ ! -d "${XNNPACK_DIR}" ]; then
  cd $XNNPACK_BASE
  ./xnnpack.sh
  cd -
else
  echo "The directory XNNPACK already exists. Skipping cloning xnnpack."
fi

# This is needed so files included by us which are used in xla can be found.
CPLUS_INCLUDE_PATH+="$ANNC/annc/service/cpu/xla/:"

# This is the location where the files included are in the main repo
# Ideally the XNNPACK repository should be downloaded and compiled by bazel so
# the directories needed should be handled automatically by bazel
CPLUS_INCLUDE_PATH+="$XNNPACK_DIR/:"
CPLUS_INCLUDE_PATH+="$XNNPACK_DIR/include/:"
CPLUS_INCLUDE_PATH+="$XNNPACK_DIR/src/:"
CPLUS_INCLUDE_PATH+="$XNNPACK_DIR/build/pthreadpool-source/include/:"
export CPLUS_INCLUDE_PATH

# Changing the value of ACTION_ENV should trigger full compilation
ACTION_ENV="baila=548"

bazel --output_user_root=./output                                              \
      build -c opt                                                             \
      --verbose_failures                                                       \
      --action_env=$ACTION_ENV                                                 \
      --define tflite_with_xnnpack=false                                       \
      annc/service/cpu:libannc.so                                              \
      --copt=-DANNC_ENABLED_OPENBLAS                                           \
      --jobs 32

