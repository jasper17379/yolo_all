# RK3588 aarch64 交叉编译工具链模板
# 用法:
#   cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchain.aarch64-rk3588.cmake \
#         -DVISION_BACKEND=RKNN \
#         -DRKNN_RT_DIR=/path/to/rknn/runtime ..
#
# 请按实际 SDK 修改 RK3588_TOOLCHAIN_ROOT

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(RK3588_TOOLCHAIN_ROOT "/opt/rk3588/aarch64-linux-gnu" CACHE PATH "交叉工具链根目录")

set(CMAKE_C_COMPILER   "${RK3588_TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${RK3588_TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-g++")

set(CMAKE_FIND_ROOT_PATH "${RK3588_TOOLCHAIN_ROOT}/aarch64-linux-gnu")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
