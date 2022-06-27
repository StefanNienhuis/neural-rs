use crate::math;

use nalgebra::{DMatrix, DVector};
use rand;
use rand_distr::{Distribution, Normal};

#[derive(bincode::Encode, bincode::Decode)]
pub struct Network {
    pub shape: Vec<usize>,

    #[bincode(with_serde)]
    pub weights: Vec<DMatrix<f64>>,

    #[bincode(with_serde)]
    pub biases: Vec<DVector<f64>>
}

impl Network {

    pub fn zeros(shape: Vec<usize>) -> Self {
        return Network {
            weights: shape[1..].iter().enumerate().map(|(i, n)| DMatrix::<f64>::zeros(*n, shape[i])).collect(),
            biases: shape[1..].iter().map(|n| DVector::<f64>::zeros(*n)).collect(),
            shape
        }
    }

    pub fn random(shape: Vec<usize>) -> Self {
        let mut network = Self::zeros(shape);
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).expect("Could not create normal distribution");

        network.weights = network.weights.iter().map(|matrix| matrix.map(|_| normal.sample(&mut rng))).collect();
        network.biases = network.biases.iter().map(|vector| vector.map(|_| normal.sample(&mut rng))).collect();

        return network;
    }

    pub fn feed_forward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut activation = DVector::from_vec(input) as DVector<f64>;

        for (weights, biases) in self.weights.iter().zip(self.biases.iter()) {
            activation = (weights * activation + biases).map(|x| math::sigmoid(x));
        }

        return activation.iter().cloned().collect();
    }

}