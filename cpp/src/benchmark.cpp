#include <torch/torch.h>
#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <torch/cuda.h>
#include <chrono>
#include "argparse.hpp"
#include <algorithm>

using namespace std::chrono;

void benchmark(std::string model_name, int n_warmup = 5, int n_iter = 5)
{
    torch::NoGradGuard no_grad;

    std::cout << "model_name: " << model_name << std::endl;
    // check that file exists
    std::ifstream file(model_name);
    if (!file.good())
    {
        std::cerr << "File not found: " << model_name << std::endl;
        std::exit(1);
    }

    auto trt_mod = torch::jit::load(model_name, torch::kCUDA);
    trt_mod.eval();
    torch::Tensor input_tensor = torch::rand({1, 3, 512, 512}).cuda();

    std::cout << "warmup" << std::endl;
    while (n_warmup--)
        auto results = trt_mod.forward({input_tensor});

    std::cout << "measuring time" << std::endl;
    std::vector<float> durations;

    while (n_iter--)
    {
        torch::cuda::synchronize();
        auto start = high_resolution_clock::now();
        auto results = trt_mod.forward({input_tensor});
        torch::cuda::synchronize();
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        durations.push_back(duration.count() * 1e-3);
    }

    float mean = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
    float sq_sum = std::inner_product(durations.begin(), durations.end(), durations.begin(), 0.0);
    float stdev = std::sqrt(sq_sum / durations.size() - mean * mean);
    std::cout << "mean: " << mean << " ms" << std::endl;
    std::cout << "std: " << stdev << " ms" << std::endl;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("benchmark");
    program.add_argument("--model")
        .help("TorchScript model path")
        .default_value("trt.ts");

    program.add_argument("--n_warmup")
        .help("Number of warmup iterations")
        .default_value(10)
        .action([](const std::string &value)
                { return std::stoi(value); });

    program.add_argument("--n_iter")
        .help("Number of benchmark iterations")
        .default_value(10)
        .action([](const std::string &value)
                { return std::stoi(value); });
    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::string model_name = program.get<std::string>("--model");
    benchmark(model_name);
    return 0;
}