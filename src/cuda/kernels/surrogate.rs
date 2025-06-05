use std::sync::{Arc, LazyLock, OnceLock};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{Ptx, compile_ptx},
};

use crate::cuda::net::Layer;

static KERNEL: LazyLock<Ptx> = LazyLock::new(|| {
    compile_ptx(
        r#"
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
    FUNCTION.get_or_init(|| {
        module(context)
            .load_function("snn_surrogate_kernel")
            .unwrap()
    })
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
        .arg(&mut layer.weighted_inputs[time_step].slice)
        .arg(&mut layer.gradients.slice)
        .arg(&n);
    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(n as u32))
            .unwrap();
    }
}
