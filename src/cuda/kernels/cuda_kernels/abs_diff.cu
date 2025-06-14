extern "C" __global__ void snn_abs_diff_kernel(
    float* a,
    float* b,
    float* c,
    size_t numel
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        a[i] = a[i] + fabsf(b[i] - c[i]);
    }
}