use std::sync::{Arc, LazyLock, OnceLock};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{Ptx, compile_ptx},
};

use crate::cuda::net::Layer;

static KERNEL: LazyLock<Ptx> =
    LazyLock::new(|| compile_ptx(include_str!("cuda_kernels/foreprop.cu")).unwrap());

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
    let n = layer.membrane_potential.len();
    assert_eq!(n, layer.weighted_inputs[time_step].len());
    assert_eq!(n, layer.spikes[time_step].len());

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
