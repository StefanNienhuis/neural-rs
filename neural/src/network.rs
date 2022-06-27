use crate::math;

use ndarray::prelude::*;
use rand::Rng;

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

        network.weights = network.weights.iter().map(|weights| weights.map(|_| rng.gen())).collect();
        network.biases = network.biases.iter().map(|biases| biases.map(|_| rng.gen())).collect();

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