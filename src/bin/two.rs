use ndarray::ArrayBase;
use ndarray::linalg::general_mat_mul as mat_mul;
use ndarray::{Array, Array2, Data, Ix2, RawData, array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr;
// /// `decay`: Sometimes reffered to as `beta`.
// /// `membrane_potential`: Sometimes reffered to as `mem`.
// fn forward(decay: f32, membrane_potential: f32)

struct Network {
    weights: Vec<Array2<f32>>,
    layers: Vec<Layer>,
}
impl Network {
    fn new(
        first: usize,
        hidden_layers: &[usize],
        weight_distribution: impl rand_distr::Distribution<f32>,
    ) -> Self {
        let (weights, layers) = std::iter::once(first)
            .chain(hidden_layers.iter().copied())
            .zip(hidden_layers.iter().copied())
            .map(|(a, b)| {
                (
                    Array::random((a, b), &weight_distribution),
                    Layer::new(0.8f32, 1f32, b),
                )
            })
            .unzip();
        Self { weights, layers }
    }
    fn forward(&mut self, mut spikes: Array2<f32>) -> Array2<f32> {
        for (layer, weights) in self.layers.iter_mut().zip(self.weights.iter()) {
            spikes = layer.forward(&spikes, &weights);
        }
        return spikes;
    }
}

struct Layer {
    decay_value: f32,
    decay: Array2<f32>,
    threshold_value: f32,
    threshold: Array2<f32>,
    membrane_potential: Array2<f32>,
}
impl Layer {
    fn new(decay: f32, threshold: f32, size: usize) -> Self {
        Self {
            decay_value: decay,
            decay: Array::from_elem((1, size), decay),
            threshold_value: threshold,
            threshold: Array::from_elem((1, size), threshold),
            membrane_potential: Array::from_elem((1, size), 0f32),
        }
    }
    // Executes the forward pass returning which neurons spiked.
    // The input is given after the weighted are applied.
    fn forward<T: RawData<Elem = f32> + Data>(
        &mut self,
        input: &Array2<f32>,
        weights: &ArrayBase<T, Ix2>,
    ) -> Array2<f32> {
        let spiked = self
            .membrane_potential
            .mapv(|m| (m > self.threshold_value) as u8 as f32);
        mat_mul(
            1f32,
            input,
            &weights,
            self.decay_value,
            &mut self.membrane_potential,
        );
        println!("bm2: {:?}", self.membrane_potential);
        self.membrane_potential =
            &self.membrane_potential - (&self.decay * &spiked * &self.threshold);
        spiked
    }
}

fn main() {
    let time_steps = 3;

    // Pre-synpatic potentials e.g. the input encodings across time
    // 1x2
    let input = vec![
        array![[0f32, 0f32]],
        array![[1f32, 0f32]],
        array![[0f32, 1f32]],
        array![[1f32, 1f32]],
        array![[0f32, 0f32]],
    ];
    assert!(input.len() >= time_steps);

    // the shape of our XOR network is 2->3->1
    let weight_distr = rand_distr::Uniform::try_from(0.7f32..=0.7f32).unwrap();
    let mut net = Network::new(2, &[3, 2], weight_distr);
    for t in 0..time_steps {
        let _s = net.forward(input[t].clone());
    }
}
