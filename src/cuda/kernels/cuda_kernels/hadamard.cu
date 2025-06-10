extern "C" __global__ void snn_hadamard_kernel(
    float* a,
    float* b,
    size_t numel
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        a[i] = a[i] * b[i];
    }
}