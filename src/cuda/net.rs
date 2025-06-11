use std::sync::Arc;

use cudarc::{
    cublas::CudaBlas,
    driver::{CudaContext, CudaStream},
};
use ndarray_rand::RandomExt;
use std::iter;

use crate::{
    PollingIterator, PollingResult,
    cuda::{CudaMatrix, kernels},
};

pub struct Network {
    pub weights: Vec<CudaMatrix>,
    pub layers: Vec<Layer>,
    pub inputs: Vec<CudaMatrix>,
    pub time_step: usize,
    pub context: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
}
impl Network {
    /// Estimate how many bytes a given structure will require.
    pub fn vram_estimate(
        first: usize,
        hidden_layers: &[usize],
        batch_size: usize,
        time_steps: usize,
    ) -> usize {
        let est = std::iter::once(first)
            .chain(hidden_layers.iter().copied())
            .zip(hidden_layers.iter().copied())
            .fold(0, |mut acc, (a, b)| {
                acc += size_of::<f32>() * a * b;
                acc += Layer::vram_estimate(b, batch_size, time_steps);
                acc
            });
        est + (0..time_steps)
            .map(|_| size_of::<f32>() * batch_size * first)
            .sum::<usize>()
    }

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
                    ),
                )
            })
            .unzip();
        Network {
            weights,
            layers,
            inputs: (0..time_steps)
                .map(|_| CudaMatrix::zeros(stream.clone(), batch_size, first).unwrap())
                .collect(),
            time_step: 0,
            context,
            stream,
        }
    }

    // TODO These lifetimes may be a little overly restrictive on `spikes`.
    /// Performs the forward pass of a training loop.
    ///
    /// - `spikes` needs to be in column-major format.
    pub fn forward(&mut self, spikes: &CudaMatrix) -> &CudaMatrix {
        self.stream
            .memcpy_dtod(&spikes.slice, &mut self.inputs[self.time_step].slice)
            .unwrap();
        let mut spikes_ref = &self.inputs[self.time_step];
        for (layer, weights) in self.layers.iter_mut().zip(self.weights.iter()) {
            spikes_ref = layer.forward(
                spikes_ref,
                weights,
                self.time_step,
                self.context.clone(),
                self.stream.clone(),
            );
        }
        self.time_step += 1;
        spikes_ref
    }

    /// Performs the backward pass of a training loop.
    ///
    /// Calculate weight updates.
    pub fn backward<'a, 'b>(
        &'a mut self,
        target_spikes: &'b [CudaMatrix],
    ) -> BackwardIterator<'a, 'b> {
        // This check should really also apply to all layers not just the output
        // layer.
        assert_eq!(
            target_spikes.len(),
            self.layers.last().unwrap().spikes.len()
        );

        BackwardIterator {
            net: self,
            target_spikes,
            inner: target_spikes.iter().enumerate().rev(),
            first: true,
        }
    }

    /// Clears stored values and resets internal timestep.
    ///
    /// This should be used before calling [`Network::forward`] when you intend
    /// to use data from these execution for training (e.g. [`Network::backward`]).
    pub fn clear(&mut self) {
        for input in self.inputs.iter_mut() {
            self.stream.memset_zeros(&mut input.slice).unwrap();
        }
        for layer in self.layers.iter_mut() {
            for weighted_inputs in layer.weighted_inputs.iter_mut() {
                self.stream
                    .memset_zeros(&mut weighted_inputs.slice)
                    .unwrap();
            }
            for spikes in layer.spikes.iter_mut() {
                self.stream.memset_zeros(&mut spikes.slice).unwrap();
            }
        }
        self.time_step = 0;
    }

    /// Update weights based of stored delta weights in each layer.
    pub fn update(&mut self, learning_rate: f32) {
        for (weights, layer) in self.weights.iter_mut().zip(self.layers.iter()) {
            crate::cuda::kernels::weight_update::run_function(
                weights,
                &layer.delta_weights,
                learning_rate,
                self.context.clone(),
                self.stream.clone(),
            );
        }
    }
}

pub struct Layer {
    // Same as cpu layer:
    pub decay: f32,
    pub threshold: f32,
    pub membrane_potential: CudaMatrix,
    pub weighted_inputs: Vec<CudaMatrix>,
    pub spikes: Vec<CudaMatrix>,
    // Different to cpu layer:
    pub cublas: Arc<CudaBlas>,
    pub gradients: CudaMatrix,
    pub errors: CudaMatrix,
    pub delta_weights: CudaMatrix,
}
impl Layer {
    /// Estimate how many bytes a given structure will require.
    pub fn vram_estimate(neurons: usize, batch_size: usize, time_steps: usize) -> usize {
        let membrane_potential = size_of::<f32>() * batch_size * neurons;
        let weight_inputs = (0..time_steps)
            .map(|_| size_of::<f32>() * batch_size * neurons)
            .sum::<usize>();
        let spikes = (0..time_steps)
            .map(|_| size_of::<f32>() * batch_size * neurons)
            .sum::<usize>();
        let gradients = size_of::<f32>() * batch_size * neurons;
        let errors = size_of::<f32>() * batch_size * neurons;
        membrane_potential + weight_inputs + spikes + gradients + errors
    }
    pub fn new(
        decay: f32,
        threshold: f32,
        neurons: usize,
        batch_size: usize,
        // CUDA specific
        cublas: Arc<CudaBlas>,
        stream: Arc<CudaStream>,
        time_steps: usize,
    ) -> Layer {
        Layer {
            cublas,
            membrane_potential: CudaMatrix::zeros(stream.clone(), batch_size, neurons).unwrap(),
            weighted_inputs: (0..time_steps)
                .map(|_| CudaMatrix::zeros(stream.clone(), batch_size, neurons).unwrap())
                .collect(),
            spikes: (0..time_steps)
                .map(|_| CudaMatrix::zeros(stream.clone(), batch_size, neurons).unwrap())
                .collect(),
            threshold,
            decay,
            gradients: CudaMatrix::zeros(stream.clone(), batch_size, neurons).unwrap(),
            errors: CudaMatrix::zeros(stream.clone(), batch_size, neurons).unwrap(),
            delta_weights: CudaMatrix::zeros(stream.clone(), batch_size, neurons).unwrap(),
        }
    }
    /// `input` and `weights` need to be in column-major format.
    pub fn forward(
        &mut self,
        input: &CudaMatrix,
        weights: &CudaMatrix,
        time_step: usize,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> &CudaMatrix {
        // stream.synchronize().unwrap();
        // let matmul_start = std::time::Instant::now();
        // TODO This does quite a bit of redundant work, would it be more
        // efficient to write a custom kernel that just does the matrix
        // multiplication?
        // Calculate weighted input for this time-step.
        crate::cuda::matmul(
            &self.cublas,
            &*input,
            weights,
            &mut self.weighted_inputs[time_step],
            false,
            false,
        );
        // stream.synchronize().unwrap();
        // println!("matmul: {:.2?}",matmul_start.elapsed());
        // let forepropfunction_start = std::time::Instant::now();
        kernels::foreprop::run_function(self, context, stream.clone(), time_step);
        // stream.synchronize().unwrap();
        // println!("forepropfunction_start: {:.2?}",forepropfunction_start.elapsed());
        &self.spikes[time_step]
    }
}

pub struct BackwardIterator<'a, 'b> {
    pub net: &'a mut Network,
    target_spikes: &'b [CudaMatrix],
    // Denoates if at the first time-step, this is useful since instead of
    // re-allocating or re-zero-ing `delta_weights` before creating
    // `BackwardIterator` we simply set `beta` to `0f32` which does it as apart
    // of `gemm`.
    first: bool,
    inner: iter::Rev<iter::Enumerate<std::slice::Iter<'b, CudaMatrix>>>,
}
impl<'a, 'b> PollingIterator for BackwardIterator<'a, 'b> {
    type Item = Vec<&'a CudaMatrix>;
    fn next(mut self) -> PollingResult<Self, Self::Item> {
        if let Some((ti, targets)) = self.inner.next() {
            // 1 when first (the first processed is the last time step), else 0.
            let beta = self.first as u8 as f32;

            let mut li = self.net.layers.len() - 1;

            let ([.., prev_layer], [layer, ..]) = self.net.layers.split_at_mut(li) else {
                unreachable!()
            };
            // Output layer.
            kernels::surrogate::run_function(
                layer,
                self.net.context.clone(),
                self.net.stream.clone(),
                ti,
            );
            kernels::output_error::run_function(
                layer,
                self.net.context.clone(),
                self.net.stream.clone(),
                ti,
                targets,
            );
            crate::cuda::gemm(
                &layer.cublas,
                &prev_layer.spikes[ti], // Previous spikes.
                &layer.errors,
                &mut layer.delta_weights,
                true,
                false,
                beta,
            );
            let [rem @ .., prev] = self.net.layers.as_mut_slice() else {
                unreachable!()
            };
            let mut remaining = rem;
            let mut delta_next = &prev.errors;
            li -= 1;

            // Hidden layers
            while li > 0 {
                let ([.., prev_layer], [layer, ..]) = remaining.split_at_mut(li) else {
                    unreachable!()
                };

                kernels::surrogate::run_function(
                    layer,
                    self.net.context.clone(),
                    self.net.stream.clone(),
                    ti,
                );
                crate::cuda::matmul(
                    &layer.cublas,
                    &delta_next,
                    &self.net.weights[li + 1],
                    &mut layer.errors,
                    false,
                    true,
                );
                kernels::hadamard::run_function(
                    &mut layer.errors,
                    &layer.gradients,
                    self.net.context.clone(),
                    self.net.stream.clone(),
                );
                crate::cuda::gemm(
                    &layer.cublas,
                    &prev_layer.spikes[ti], // Previous spikes.
                    &layer.errors,
                    &mut layer.delta_weights,
                    true,
                    false,
                    beta,
                );

                let [rem @ .., prev] = remaining else {
                    unreachable!()
                };
                remaining = rem;
                delta_next = &prev.errors;
                li -= 1;
            }

            // Input layer
            let layer = &mut remaining[li];
            kernels::surrogate::run_function(
                layer,
                self.net.context.clone(),
                self.net.stream.clone(),
                ti,
            );
            crate::cuda::matmul(
                &layer.cublas,
                &delta_next,
                &self.net.weights[li + 1],
                &mut layer.errors,
                false,
                true,
            );
            kernels::hadamard::run_function(
                &mut layer.errors,
                &layer.gradients,
                self.net.context.clone(),
                self.net.stream.clone(),
            );
            crate::cuda::gemm(
                &layer.cublas,
                &self.net.inputs[ti], // Previous spikes.
                &layer.errors,
                &mut layer.delta_weights,
                true,
                false,
                beta,
            );
            self.first = false;
            PollingResult::Incomplete(self)
        } else {
            // Divide weight updates by time steps.
            let n = self.target_spikes.iter().map(|t| t.len()).sum::<usize>() as f32;
            for layer in self.net.layers.iter_mut() {
                kernels::div::run_function(
                    &mut layer.delta_weights,
                    n,
                    self.net.context.clone(),
                    self.net.stream.clone(),
                );
            }

            // Return weight updates.
            PollingResult::Complete(
                self.net
                    .layers
                    .iter()
                    .map(|l| &l.delta_weights)
                    .collect::<Vec<_>>(),
            )
        }
    }
}
