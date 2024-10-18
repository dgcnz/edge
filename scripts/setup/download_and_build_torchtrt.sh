# check if third-party/TensorRT exists and clone if not
if [ -d "third-party/TensorRT" ]; then
    echo "[INFO]: TensorRT already exists in third-party"
else
    echo "[INFO]: Downloading TensorRT from NVIDIA to third-party"
    mkdir -p third-party
    git clone https://github.com/pytorch/TensorRT third-party/TensorRT
fi


cd third-party/TensorRT
echo "[INFO]: Building TorchTRT"

# if gcc > 13, then we need to prepend #include <cstdint> to core/util/Exception.h
# check if the second line to core/util/Exception.h is #include <cstdint>
# if not, add #include <cstdint> to the second line of core/util/Exception.h

if ! grep -q "#include <cstdint>" core/util/Exception.h; then
    echo "[DEUBG] Adding #include <cstdint> to core/util/Exception.h"
    sed -i '2i #include <cstdint>' core/util/Exception.h
fi

# Instead of downloading TensorRT from NVIDIA
# we will use the precompiled libraries installed by pip 
# and the header files from the cloned repository
cmake -Bbuild \
    -DTensorRT_INCLUDE_DIR=../NVIDIATensorRT/include/ \
    -DTensorRT_LIBRARY=$CONDA_PREFIX/lib/python3.10/site-packages/tensorrt_libs/libnvinfer.so.10 \
    -DTensorRT_nvinfer_plugin_LIBRARY=$CONDA_PREFIX/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_plugin.so.10  \
    -DTorch_DIR=../libtorch/share/cmake/Torch \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MODULE_PATH=cmake/Modules

echo "[INFO]: Compiling TorchTRT"

cmake --build build --config Release