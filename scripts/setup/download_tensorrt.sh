# Check if tensorrt exists and download if not
if [ -d "third-party/NVIDIATensorRT" ]; then
    echo "NVIDIATensorRT already exists in third-party"
    exit 0
fi

version=$(pip show tensorrt | grep Version: | awk '{print $2}')
echo "The installed tensorrt version is $version"

# Check if the prefix is 10.3
if [[ $version == 10.3* ]]; then
    echo "Downloading TensorRT from NVIDIA to third-party"
    mkdir -p third-party
    git clone --depth 1 --branch release/10.3  https://github.com/NVIDIA/TensorRT.git third-party/NVIDIATensorRT
else
    >&2 echo "WARNING: The installed tensorrt by pip is not 10.3. The compilation scripts may not work, adapt them to your version if necessary."
fi
