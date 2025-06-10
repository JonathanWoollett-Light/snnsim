extern "C" __global__ void snn_output_error_kernel(
    float* spikes,
    float* targets,
    float* gradients,
    float* errors,
    size_t numel
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        errors[i] = (spikes[i] - targets[i]) * gradients[i];
    }
}