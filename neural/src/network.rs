use crate::{functions::*};

use nalgebra::{DMatrix, DVector};
use rand::thread_rng;

#[derive(bincode::Encode, bincode::Decode)]
pub struct Network {
    pub(crate) layers: Vec<Layer>,

    #[bincode(with_serde)]
    pub weights: Vec<DMatrix<f64>>,

    #[bincode(with_serde)]
    pub biases: Vec<DVector<f64>>
}

#[derive(bincode::Encode, bincode::Decode)]
pub struct Layer {
    pub activation_function: ActivationFunction,
    pub size: usize
}

impl Network {

    pub fn new() -> Self {
        return Self {
            layers: vec![],
            weights: vec![],
            biases: vec![]
        }
    }

    pub fn add_layer(&mut self, size: usize, activation_function: ActivationFunction) {
        if matches!(activation_function, ActivationFunction::Input) && self.layers.len() != 0 ||
            !matches!(activation_function, ActivationFunction::Input) && self.layers.len() == 0 {
            panic!("Input layer should be first")
        }

        if !matches!(activation_function, ActivationFunction::Input) {
            let mut rng = thread_rng();
            let previous_layer = self.layers.last().expect("No previous layer").size;
            let weights = DMatrix::<f64>::zeros(size, previous_layer);

            self.weights.push(weights.map(|_| activation_function.initialize_weight(previous_layer, &mut rng)));
            self.biases.push(DVector::<f64>::zeros(size));
        }

        self.layers.push(Layer {
            size, activation_function
        });
    }

    pub fn shape(&self) -> Vec<usize> {
        return self.layers.iter().map(|l| l.size).collect();
    }

    pub fn feed_forward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut activation = DVector::from_vec(input) as DVector<f64>;

        for (m, (weights, biases)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            activation = (weights * activation + biases).map(|x| self.layers[m + 1].activation_function.function(x));
        }

        return activation.iter().cloned().collect();
    }

}