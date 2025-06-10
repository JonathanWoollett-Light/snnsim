extern "C" __global__ void snn_weight_update_kernel(
    float* a,
    float* b,
    float c,
    size_t numel
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        a[i] = a[i] - (b[i] * c);
    }
}