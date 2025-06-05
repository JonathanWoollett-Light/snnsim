use std::sync::{Arc, LazyLock, OnceLock};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    },
    nvrtc::{Ptx, compile_ptx},
};

use crate::cuda::{CudaMatrix, net::Layer};

static KERNEL: LazyLock<Ptx> = LazyLock::new(|| {
    compile_ptx(
        r#"
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
            .load_function("snn_output_error_kernel")
            .unwrap()
    })
}

pub fn run_function(
    layer: &mut Layer,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    time_step: usize,
    target_spikes: &CudaMatrix,
) {
    let n = layer.neurons * layer.batch_size;
    let mut builder = stream.launch_builder(function(context));
    builder
        .arg(&mut layer.spikes[time_step].slice)
        .arg(&target_spikes.slice)
        .arg(&layer.gradients.slice)
        .arg(&mut layer.errors.slice)
        .arg(&n);
    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(n as u32))
            .unwrap();
    }
}
