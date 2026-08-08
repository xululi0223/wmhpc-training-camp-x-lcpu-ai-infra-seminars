#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err_ = (call);                                  \
        if (err_ != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",         \
                    cudaGetErrorName(err_), __FILE__, __LINE__,     \
                    cudaGetErrorString(err_));                      \
            exit(1);                                                \
        }                                                           \
    } while (0)

#define CUDA_CHECK_KERNEL()                         \
    do {                                            \
        CUDA_CHECK(cudaGetLastError());             \
        CUDA_CHECK(cudaDeviceSynchronize());        \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() {CUDA_CHECK(cudaEventRecord(start_));}
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

__global__ void saxpy_kernel(const float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 2.0f * x[idx] + y[idx];
    }
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    if (n == 0) {
        printf("SUM=0, n=0\n");
        return 0;
    }

    size_t bytes = n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = ((i % 2048) - 1024) * 0.5f;
        h_b[i] = (i % 1024) - 512;
    }

    GpuTimer timer;

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    timer.start();
    saxpy_kernel<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_CHECK_KERNEL();
    float ms = timer.stop_ms();

    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += h_b[i];
    }

    printf("SUM=%lf, n=%d, time=%.3f ms\n", sum, n, ms);
    return 0;
}
