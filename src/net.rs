use std::f32::consts::PI;

use ndarray::{Array, Array2, Data, Ix2, RawData};
use ndarray::{Array3, ArrayView2};
use ndarray::{ArrayBase, Axis};
use ndarray_rand::RandomExt;

// use ndarray::Array1;
// /// `decay`: Sometimes referred to as `beta`.
// /// `membrane_potential`: Sometimes referred to as `mem`.
// fn forward(decay: f32, membrane_potential: f32)

pub struct Network {
    // Stuff needed for foreprop:
    pub weights: Vec<Array2<f32>>,
    pub layers: Vec<Layer>,
    // Stuff only needed for backprop:
    pub inputs: Vec<Array2<f32>>,
}
impl Network {
    /// `batch_size`: Number of samples in each training batch.
    pub fn new(
        first: usize,
        hidden_layers: &[usize],
        weight_distribution: impl ndarray_rand::rand_distr::Distribution<f32>,
        batch_size: usize,
    ) -> Self {
        let (weights, layers) = std::iter::once(first)
            .chain(hidden_layers.iter().copied())
            .zip(hidden_layers.iter().copied())
            .map(|(a, b)| {
                (
                    Array::random((a, b), &weight_distribution),
                    Layer::new(
                        crate::DEFAULT_DECAY,
                        crate::DEFAULT_THRESHOLD,
                        b,
                        batch_size,
                    ),
                )
            })
            .unzip();
        Self {
            weights,
            layers,
            inputs: Vec::new(),
        }
    }
    /// Pass data through the network.
    ///
    /// `spikes`: `[samples x features]`.
    ///
    /// Returns `[samples x output spikes]`.
    pub fn forward(&mut self, spikes: Array2<f32>) -> ArrayView2<f32> {
        self.inputs.push(spikes);

        let mut spikes_view = self.inputs.last().unwrap().view();
        for (layer, weights) in self.layers.iter_mut().zip(self.weights.iter()) {
            spikes_view = layer.forward(spikes_view, weights);
        }
        spikes_view
    }

    /// Calculate weight updates.
    ///
    /// - `target_spikes`: `[time steps x samples x features]`
    pub fn backward<'a, 'b>(
        &'a mut self,
        target_spikes: &'b Array3<f32>,
    ) -> BackwardIterator<'a, 'b> {
        // This check should really also apply to all layers not just the output
        // layer.
        assert_eq!(
            target_spikes.dim().0,
            self.layers.last().unwrap().spikes.len()
        );

        // When updating `delta_weights` both `output_error` and
        // `output_previous_spikes` are matrices of shapes 1xA and 1xB,
        // we want the result to be AxB (this is the same as `numpy.outer`).
        let delta_weights = self
            .weights
            .iter()
            .map(|w| Array2::<f32>::zeros(w.raw_dim()))
            .collect::<Vec<_>>();

        BackwardIterator {
            net: self,
            target_spikes,
            delta_weights,
            inner: target_spikes.axis_iter(Axis(0)).enumerate().rev(),
        }
    }

    /// Update weights.
    pub fn update(&mut self, learning_rate: f32, delta_weights: Vec<Array2<f32>>) {
        // Reset stored data.
        self.inputs = Vec::new();
        for layer in self.layers.iter_mut() {
            layer.weighted_inputs = Vec::new();
            layer.spikes = Vec::new();
        }

        // Update weights.
        for (weights, delta_weights) in self.weights.iter_mut().zip(delta_weights.into_iter()) {
            // TODO Surely we don't need to clone `weights` here?
            *weights = weights.clone() - (learning_rate * delta_weights);
        }
    }
}

/// Surrogate gradient (arctan derivative).
fn surrogate_gradient(membrane_potentials: &Array2<f32>) -> Array2<f32> {
    1f32 / (1f32 + (PI * membrane_potentials).pow2())
}

pub struct Layer {
    // Stuff needed for foreprop:
    pub decay_value: f32,
    pub decay: Array2<f32>,
    pub threshold_value: f32,
    pub threshold: Array2<f32>,
    /// `[samples x neurons]`
    pub membrane_potential: Array2<f32>,
    // Stuff only needed for backprop:
    pub weighted_inputs: Vec<Array2<f32>>, // Stores the weighted inputs for previous time-steps.
    pub spikes: Vec<Array2<f32>>,
}
impl Layer {
    /// - `size`: Number of neurons in the layer.
    /// - `batch`: Number of samples in each training batch.
    fn new(decay: f32, threshold: f32, neurons: usize, batch_size: usize) -> Self {
        Self {
            decay_value: decay,
            decay: Array::from_elem((batch_size, neurons), decay),
            threshold_value: threshold,
            threshold: Array::from_elem((batch_size, neurons), threshold),
            membrane_potential: Array::from_elem((batch_size, neurons), 0f32),
            weighted_inputs: Vec::new(),
            spikes: Vec::new(),
        }
    }
    /// Executes the forward pass returning which neurons spiked.
    /// The input is given after the weighted are applied.
    ///
    /// - `input`: `[samples x features]`
    /// - `weights`: `[features x neurons]`
    fn forward<'a, T: RawData<Elem = f32> + Data>(
        &'a mut self,
        input: ArrayView2<f32>,
        weights: &ArrayBase<T, Ix2>,
    ) -> ArrayView2<'a, f32> {
        // Apply input and decay.
        let spiked = self
            .membrane_potential
            .mapv(|m| (m > self.threshold_value) as u8 as f32);

        let weighted_inputs = input.dot(weights);
        self.membrane_potential = &weighted_inputs + self.decay_value * &self.membrane_potential;

        // Store weighted inputs for backprop.
        self.weighted_inputs.push(weighted_inputs);

        // Apply threshold reset.
        self.membrane_potential =
            &self.membrane_potential - (&self.decay * &spiked * &self.threshold);
        // println!("m: {:?}",self.membrane_potential.as_slice().unwrap());
        self.spikes.push(spiked.clone());
        self.spikes.last().unwrap().view()
    }
}

use crate::{PollingIterator, PollingResult};
use std::iter;

pub struct BackwardIterator<'a, 'b> {
    pub net: &'a mut Network,
    target_spikes: &'b Array3<f32>,
    pub delta_weights: Vec<Array2<f32>>,
    inner: iter::Rev<iter::Enumerate<ndarray::iter::AxisIter<'b, f32, ndarray::Dim<[usize; 2]>>>>,
}
impl<'a, 'b> PollingIterator for BackwardIterator<'a, 'b> {
    type Item = Vec<Array2<f32>>;
    fn next(mut self) -> PollingResult<Self, Self::Item> {
        if let Some((ti, targets)) = self.inner.next() {
            let mut li = self.net.layers.len() - 1;

            // Output layer.
            let output_grad = surrogate_gradient(&self.net.layers[li].weighted_inputs[ti]);
            let output_error = (&self.net.layers[li].spikes[ti] - &targets) * output_grad;
            let output_previous_spikes = &self.net.layers[li - 1].spikes[ti];
            self.delta_weights[li] += &output_previous_spikes.t().dot(&output_error);
            let mut delta_next = output_error;
            // println!("out delta: {:?}", self.delta_weights[li]);
            li -= 1;

            // Hidden layers.
            while li > 0 {
                let grad = surrogate_gradient(&self.net.layers[li].weighted_inputs[ti]);
                let error = delta_next.dot(&self.net.weights[li + 1].t()) * grad;
                let previous_spikes = &self.net.layers[li - 1].spikes[ti];
                self.delta_weights[li] += &previous_spikes.t().dot(&error);
                delta_next = error;
                li -= 1;
            }

            // Input layer.
            let grad = surrogate_gradient(&self.net.layers[li].weighted_inputs[ti]);
            let error = delta_next.dot(&self.net.weights[li + 1].t()) * grad;
            let previous_spikes = &self.net.inputs[ti];
            self.delta_weights[li] += &previous_spikes.t().dot(&error);
            // println!("in delta: {:?}", self.delta_weights[li]);

            PollingResult::Incomplete(self)
        } else {
            // Divide weight updates by time steps.
            for delta_weights in self.delta_weights.iter_mut() {
                *delta_weights /= self.target_spikes.len() as f32;
            }

            // Return weight updates.
            PollingResult::Complete(self.delta_weights)
        }
    }
}
