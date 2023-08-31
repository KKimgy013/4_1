#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system/cuda/execution_policy.h>
#include <chrono>

#define NUM_STEPS 1000000000

struct AtomicAdd{
    template <typename T>
    __device__ void operator()(T* addr, T val) const{
        atomicAdd(addr, val);
    }
};

struct ComputePi{
    double step;

    __device__ double operator()(int i) const{
        double x = (i + 0.5) * step;
        double fx = 4.0 / (1.0 + (x * x));
        return fx * step;
    }
};

int main(){
    double step = 1.0 / (double)NUM_STEPS;
    thrust::device_vector<double> sum_d(1, 0.0);

    ComputePi computePi{step};

    auto start_time = std::chrono::steady_clock::now();
    double pi = thrust::transform_reduce(thrust::cuda::par,
                                         thrust::counting_iterator<int>(0),
                                         thrust::counting_iterator<int>(NUM_STEPS + 1),
                                         computePi,
                                         0.0,
                                         thrust::plus<double>());

    cudaDeviceSynchronize(); // 동기화를 위해 추가
    auto end_time = std::chrono::steady_clock::now();

    cudaError_t cuda_error = cudaGetLastError(); // CUDA 에러 체크
    if (cuda_error != cudaSuccess){
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    printf("Execution Time: %.10lf sec\n", elapsed_seconds.count());
    printf("pi=%.10lf\n", pi);

    return 0;
}