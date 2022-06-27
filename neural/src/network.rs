use crate::math;

use ndarray::prelude::*;
use rand;
use rand_distr::{Distribution, Normal};

#[derive(bincode::Encode, bincode::Decode)]
pub struct Network {
    pub shape: Vec<usize>,

    #[bincode(with_serde)]
    pub weights: Vec<Array2<f64>>,

    #[bincode(with_serde)]
    pub biases: Vec<Array1<f64>>
}

impl Network {

    pub fn zeros(shape: Vec<usize>) -> Network {
        let weights: Vec<Array2<f64>> = shape[1..].iter().enumerate().map(|(i, n)| Array2::zeros((*n, shape[i]))).collect();
        let biases: Vec<Array1<f64>> = shape[1..].iter().map(|n| Array1::zeros(*n)).collect();

        return Network {
            shape, weights, biases
        }
    }

    pub fn random(shape: Vec<usize>) -> Network {
        let mut network = Network::zeros(shape);
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).expect("Could not create normal distribution");

        network.weights = network.weights.iter().map(|weights| weights.map(|_| normal.sample(&mut rng))).collect();
        network.biases = network.biases.iter().map(|biases| biases.map(|_| normal.sample(&mut rng))).collect();

        return network;
    }

    pub fn feed_forward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut activation = Array1::from_vec(input) as Array1<f64>;

        for (weights, biases) in self.weights.iter().zip(self.biases.iter()) {
            activation = (weights.dot(&activation) + biases).map(|x| math::sigmoid(*x));
        }

        return activation.to_vec();
    }

}