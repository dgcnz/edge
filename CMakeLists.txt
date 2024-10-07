cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(example-app)
# set(torchtrt_DIR /teamspace/studios/this_studio/third-party/torch_tensorrt)  
# set(TensorRT_ROOT /teamspace/studios/this_studio/third-party/torch_tensorrt)  


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
include_directories(
  ${PROJECT_SOURCE_DIR}/third-party/torch_tensorrt/include
    )
# link_directories(/teamspace/studios/this_studio/third-party/torch_tensorrt)
target_link_libraries(example-app -Wl,--no-as-needed ${PROJECT_SOURCE_DIR}/third-party/torch_tensorrt/lib/libtorchtrt.so) #${PROJECT_SOURCE_DIR}/third-party/torch_tensorrt/lib/libtorchtrt_runtime.so ${PROJECT_SOURCE_DIR}/third-party/torch_tensorrt/lib/libtorchtrt_plugins.so)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries (example-app 
#                                        ${PROJECT_SOURCE_DIR}/third-party/libtorch/lib/libtorch.so
#                                        ${PROJECT_SOURCE_DIR}/third-party/libtorch/lib/libtorch_cuda.so
#                                        ${PROJECT_SOURCE_DIR}/third-party/libtorch/lib/libtorch_cpu.so
                                       ${PROJECT_SOURCE_DIR}/third-party/libtorch/lib/libtorch_global_deps.so
#                                        ${PROJECT_SOURCE_DIR}/third-party/libtorch/lib/libbackend_with_compiler.so
#                                        ${PROJECT_SOURCE_DIR}/third-party/libtorch/lib/libc10.so
#                                        ${PROJECT_SOURCE_DIR}/third-party/libtorch/lib/libc10_cuda.so
                                        )

set_property(TARGET example-app PROPERTY CXX_STANDARD 17)



# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)