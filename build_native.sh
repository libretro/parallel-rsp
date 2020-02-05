#!/bin/bash

mkdir -p build-native
cd build-native
cmake .. -DCMAKE_BUILD_TYPE=Debug -DPARALLEL_RSP_DEBUG_JIT=OFF
cmake --build . --parallel
cd ..
