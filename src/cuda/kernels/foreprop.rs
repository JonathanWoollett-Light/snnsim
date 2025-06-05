use std::sync::{Arc, LazyLock, OnceLock};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{Ptx, compile_ptx},
};

use crate::cuda::net::Layer;

static KERNEL: LazyLock<Ptx> = LazyLock::new(|| {
    compile_ptx(
        r#"
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
}"#,
    )
    .unwrap()
});

fn module(context: Arc<CudaContext>) -> Arc<CudaModule> {
    static FOREPROP_MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();
    FOREPROP_MODULE
        .get_or_init(|| context.load_module((*KERNEL).clone()).unwrap())
        .clone()
}
fn function<'a>(context: Arc<CudaContext>) -> &'a CudaFunction {
    static FUNCTION: OnceLock<CudaFunction> = OnceLock::new();
    FUNCTION.get_or_init(|| module(context).load_function("snn_forward_kernel").unwrap())
}

pub fn run_function(
    layer: &mut Layer,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    time_step: usize,
) {
    let n = layer.neurons * layer.batch_size;
    let mut builder = stream.launch_builder(function(context));
    builder
        .arg(&mut layer.membrane_potential.slice)
        .arg(&layer.weighted_inputs[time_step].slice)
        .arg(&mut layer.spikes[time_step].slice)
        .arg(&layer.threshold)
        .arg(&layer.decay)
        .arg(&n);
    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(n as u32))
            .unwrap();
    }
}
