use std::sync::{Arc, LazyLock, OnceLock};

use cudarc::driver::PushKernelArg;
use cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t},
    driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig},
    nvrtc::{Ptx, compile_ptx},
};
use ndarray_rand::RandomExt;

pub struct Network {
    pub weights: Vec<CudaSlice<f32>>,
    pub layers: Vec<Layer>,
    pub inputs: Vec<CudaSlice<f32>>,
    pub time_step: usize,
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
}
impl Network {
    pub fn new(
        first: usize,
        hidden_layers: &[usize],
        weight_distribution: impl ndarray_rand::rand_distr::Distribution<f32>,
        batch_size: usize,
        time_steps: usize,
    ) -> Network {
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let cublas = Arc::new(CudaBlas::new(stream.clone()).unwrap());

        let (weights, layers) = std::iter::once(first)
            .chain(hidden_layers.iter().copied())
            .zip(hidden_layers.iter().copied())
            .map(|(a, b)| {
                let weights = ndarray::Array::random((a, b), &weight_distribution);
                (
                    stream.memcpy_stod(weights.as_slice().unwrap()).unwrap(),
                    Layer::new(
                        crate::DEFAULT_DECAY,
                        crate::DEFAULT_THRESHOLD,
                        b,
                        batch_size,
                        cublas.clone(),
                        stream.clone(),
                        time_steps,
                        a,
                    ),
                )
            })
            .unzip();
        Network {
            weights,
            layers,
            inputs: Vec::new(),
            time_step: 0,
            context,
            stream,
        }
    }
    /// `spikes` needs to be in column-major format.
    pub fn forward(&mut self, mut spikes: CudaSlice<f32>) -> CudaSlice<f32> {
        self.inputs.push(spikes.clone());
        for (layer, weights) in self.layers.iter_mut().zip(self.weights.iter()) {
            spikes = layer.forward(
                spikes,
                weights,
                self.time_step,
                self.context.clone(),
                self.stream.clone(),
            );
        }
        self.time_step += 1;
        spikes
    }
}

static FOREWORD_KERNEL: LazyLock<Ptx> = LazyLock::new(|| {
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

static FOREPROP_MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();
static FOREPROP_FUNCTION: OnceLock<CudaFunction> = OnceLock::new();

fn foreprop_module(context: Arc<CudaContext>) -> Arc<CudaModule> {
    FOREPROP_MODULE
        .get_or_init(|| context.load_module((*FOREWORD_KERNEL).clone()).unwrap())
        .clone()
}
fn foreprop_function<'a>(context: Arc<CudaContext>) -> &'a CudaFunction {
    FOREPROP_FUNCTION.get_or_init(|| {
        foreprop_module(context)
            .load_function("snn_forward_kernel")
            .unwrap()
    })
}

pub struct Layer {
    pub cublas: Arc<CudaBlas>,
    pub membrane_potential: CudaSlice<f32>,
    pub weighted_inputs: Vec<CudaSlice<f32>>,
    pub spikes: Vec<CudaSlice<f32>>,
    pub batch_size: usize,
    pub input_features: usize,
    pub neurons: usize,
    pub threshold: f32,
    pub decay: f32,
}
impl Layer {
    pub fn new(
        decay: f32,
        threshold: f32,
        neurons: usize,
        batch_size: usize,
        // CUDA specific
        cublas: Arc<CudaBlas>,
        stream: Arc<CudaStream>,
        time_steps: usize,
        input_features: usize,
    ) -> Layer {
        Layer {
            cublas,
            membrane_potential: stream.alloc_zeros::<f32>(neurons * batch_size).unwrap(),
            weighted_inputs: (0..time_steps)
                .map(|_| stream.alloc_zeros::<f32>(neurons * batch_size).unwrap())
                .collect(),
            spikes: (0..time_steps)
                .map(|_| stream.alloc_zeros::<f32>(neurons * batch_size).unwrap())
                .collect(),
            batch_size,
            input_features,
            neurons,
            threshold,
            decay,
        }
    }
    /// `input` and `weights` need to be in column-major format.
    pub fn forward(
        &mut self,
        input: CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        time_step: usize,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> CudaSlice<f32> {
        let m = self.batch_size as i32;
        let n = self.neurons as i32;
        let k = self.input_features as i32;

        // TODO It would be better to store everything column major to avoid
        // needing the transposes here.
        // TODO This does quite a bit of redundant work, would it be more
        // efficient to write a custom kernel that just does the matrix
        // multiplication?
        // Calculate weighted input for this time-step.
        unsafe {
            self.cublas
                .gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m,
                        n,
                        k,
                        alpha: 1f32,
                        lda: m,
                        ldb: k,
                        beta: 0f32,
                        ldc: m,
                    },
                    &input,
                    weights,
                    &mut self.weighted_inputs[time_step],
                )
                .unwrap()
        }

        let mut builder = stream.launch_builder(foreprop_function(context));
        builder
            .arg(&mut self.membrane_potential)
            .arg(&self.weighted_inputs[time_step])
            .arg(&mut self.spikes[time_step])
            .arg(&self.threshold)
            .arg(&self.decay)
            .arg(&self.neurons);
        unsafe {
            builder
                .launch(LaunchConfig::for_num_elems(self.neurons as u32))
                .unwrap();
        }
        self.spikes[time_step].clone()
    }
}
