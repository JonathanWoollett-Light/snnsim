use std::sync::{Arc, LazyLock, OnceLock};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{Ptx, compile_ptx},
};

use crate::cuda::CudaMatrix;

static KERNEL: LazyLock<Ptx> =
    LazyLock::new(|| compile_ptx(include_str!("cuda_kernels/weight_update.cu")).unwrap());

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
            .load_function("snn_weight_update_kernel")
            .unwrap()
    })
}

pub fn run_function(
    a: &mut CudaMatrix,
    b: &CudaMatrix,
    c: f32,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) {
    let n = a.len();
    let mut builder = stream.launch_builder(function(context));
    builder.arg(&mut a.slice).arg(&b.slice).arg(&c).arg(&n);
    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(n as u32))
            .unwrap();
    }
}
