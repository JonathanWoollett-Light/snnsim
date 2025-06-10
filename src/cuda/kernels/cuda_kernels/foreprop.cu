extern "C" __global__ void snn_forward_kernel(
    float* membrane_potential,
    const float* weighted_inputs,
    float* spiked_output,
    float threshold,
    float decay,
    size_t numel
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        // Spike calculation
        float spiked = (membrane_potential[i] > threshold) ? 1.0f : 0.0f;
        spiked_output[i] = spiked;

        // Membrane potential update
        float new_potential = weighted_inputs[i] + 
                            decay * membrane_potential[i] - 
                            decay * spiked * threshold;
        
        membrane_potential[i] = new_potential;
    }
}