#include <torch/torch.h>
#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <torch/cuda.h>
#include <chrono>

using namespace std::chrono;

int main() {
    auto trt_mod = torch::jit::load("trt.ts");
    torch::Tensor input_tensor = torch::rand({1, 3, 512, 512}).cuda();
    
    std::cout << "warmup" << std::endl;
    for (auto i = 0; i < 5; ++i) 
        auto results = trt_mod.forward({input_tensor});
    
    std::cout << "measuring time" << std::endl;
    for (auto i = 0; i < 5; ++i) 
    {
        torch::cuda::synchronize();
        auto start = high_resolution_clock::now();
        auto results = trt_mod.forward({input_tensor});
        torch::cuda::synchronize();
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        std::cout  << duration.count() << std::endl;
    }
    return 0;
}