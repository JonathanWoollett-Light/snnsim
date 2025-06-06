use std::sync::{Arc, LazyLock, OnceLock};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{Ptx, compile_ptx},
};

use crate::cuda::CudaMatrix;

static KERNEL: LazyLock<Ptx> = LazyLock::new(|| {
    compile_ptx(
        r#"
extern "C" __global__ void snn_div_kernel(
    float* a,
    float b,
    size_t numel
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        a[i] = a[i] / b;
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
    FUNCTION.get_or_init(|| module(context).load_function("snn_div_kernel").unwrap())
}

pub fn run_function(
    a: &mut CudaMatrix,
    b: f32,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) {
    let n = a.len();
    let mut builder = stream.launch_builder(function(context));
    builder.arg(&mut a.slice).arg(&b).arg(&n);
    unsafe {
        builder
            .launch(LaunchConfig::for_num_elems(n as u32))
            .unwrap();
    }
}
