extern "C" __global__ void snn_surrogate_kernel(
    float* membrane_potential,
    float* gradient,
    size_t numel
) {
    #define PI 3.141592654f
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float x = PI * membrane_potential[i];
        gradient[i] = 1.0f / (1.0f + (x * x));
    }
}