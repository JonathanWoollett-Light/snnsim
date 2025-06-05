use std::sync::Arc;

use cudarc::{
    cublas::CudaBlas,
    driver::{CudaContext, CudaStream},
};
use ndarray_rand::RandomExt;

use crate::cuda::{CudaMatrix, kernels};

pub struct Network {
    pub weights: Vec<CudaMatrix>,
    pub layers: Vec<Layer>,
    pub inputs: Vec<CudaMatrix>,
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
                    CudaMatrix::from_ndarray(stream.clone(), weights),
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
    pub fn forward(&mut self, mut spikes: CudaMatrix) -> CudaMatrix {
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

    /// Calculate weight updates.
    pub fn backward(&mut self, target_spikes: &[CudaMatrix]) -> Vec<CudaMatrix> {
        // This check should really also apply to all layers not just the output
        // layer.
        assert_eq!(
            target_spikes.len(),
            self.layers.last().unwrap().spikes.len()
        );

        let mut delta_weights = self
            .weights
            .iter()
            .map(|w| CudaMatrix::zeros(self.stream.clone(), w.rows, w.columns))
            .collect::<Vec<_>>();
        for (ti, targets) in target_spikes.iter().enumerate() {
            let mut li = self.layers.len() - 1;

            // Output layer.
            kernels::surrogate::run_function(
                &mut self.layers[li],
                self.context.clone(),
                self.stream.clone(),
                ti,
            );
            kernels::output_error::run_function(
                &mut self.layers[li],
                self.context.clone(),
                self.stream.clone(),
                ti,
                targets,
            );
            let output_previous_spikes = &self.layers[li - 1].spikes[ti];
            crate::cuda::gemm(
                &self.layers[li].cublas,
                output_previous_spikes,
                &self.layers[li].errors,
                &mut delta_weights[li],
                true,
                false,
                1f32,
            );
            let mut delta_next = self.layers[li].errors.clone();
            li -= 1;

            // Hidden layers
            while li > 0 {
                let layer = &mut self.layers[li];
                kernels::surrogate::run_function(
                    layer,
                    self.context.clone(),
                    self.stream.clone(),
                    ti,
                );
                crate::cuda::matmul(
                    &layer.cublas,
                    &delta_next,
                    &self.weights[li + 1],
                    &mut layer.errors,
                    false,
                    true,
                );
                kernels::hadamard::run_function(
                    &mut layer.errors,
                    &layer.gradients,
                    self.context.clone(),
                    self.stream.clone(),
                );
                let previous_spikes = &self.layers[li - 1].spikes[ti];
                crate::cuda::gemm(
                    &self.layers[li].cublas,
                    previous_spikes,
                    &self.layers[li].errors,
                    &mut delta_weights[li],
                    true,
                    false,
                    1f32,
                );
                delta_next = self.layers[li].errors.clone();
                li -= 1;
            }

            // Input layer
            let layer = &mut self.layers[li];
            kernels::surrogate::run_function(layer, self.context.clone(), self.stream.clone(), ti);
            crate::cuda::matmul(
                &layer.cublas,
                &delta_next,
                &self.weights[li + 1],
                &mut layer.errors,
                false,
                true,
            );
            kernels::hadamard::run_function(
                &mut layer.errors,
                &layer.gradients,
                self.context.clone(),
                self.stream.clone(),
            );
            let previous_spikes = &self.inputs[ti];
            crate::cuda::gemm(
                &layer.cublas,
                previous_spikes,
                &layer.errors,
                &mut delta_weights[li],
                true,
                false,
                1f32,
            );
        }

        // Divide weight updates by time steps.
        for delta_weights in delta_weights.iter_mut() {
            kernels::div::run_function(
                delta_weights,
                target_spikes.len() as f32,
                self.context.clone(),
                self.stream.clone(),
            );
        }

        delta_weights
    }
}

pub struct Layer {
    pub cublas: Arc<CudaBlas>,
    pub membrane_potential: CudaMatrix,
    pub weighted_inputs: Vec<CudaMatrix>,
    pub spikes: Vec<CudaMatrix>,
    pub batch_size: usize,
    pub input_features: usize,
    pub neurons: usize,
    pub threshold: f32,
    pub decay: f32,
    // Backprop stuff
    pub gradients: CudaMatrix,
    pub errors: CudaMatrix,
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
            membrane_potential: CudaMatrix::zeros(stream.clone(), batch_size, neurons),
            weighted_inputs: (0..time_steps)
                .map(|_| CudaMatrix::zeros(stream.clone(), batch_size, neurons))
                .collect(),
            spikes: (0..time_steps)
                .map(|_| CudaMatrix::zeros(stream.clone(), batch_size, neurons))
                .collect(),
            batch_size,
            input_features,
            neurons,
            threshold,
            decay,
            gradients: CudaMatrix::zeros(stream.clone(), batch_size, neurons),
            errors: CudaMatrix::zeros(stream.clone(), batch_size, neurons),
        }
    }
    /// `input` and `weights` need to be in column-major format.
    pub fn forward(
        &mut self,
        input: CudaMatrix,
        weights: &CudaMatrix,
        time_step: usize,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> CudaMatrix {
        // TODO This does quite a bit of redundant work, would it be more
        // efficient to write a custom kernel that just does the matrix
        // multiplication?
        // Calculate weighted input for this time-step.
        crate::cuda::matmul(
            &self.cublas,
            &input,
            weights,
            &mut self.weighted_inputs[time_step],
            false,
            false,
        );
        kernels::foreprop::run_function(self, context, stream, time_step);
        self.spikes[time_step].clone()
    }
}
