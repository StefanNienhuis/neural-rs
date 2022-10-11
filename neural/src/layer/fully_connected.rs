use crate::{Layer, Float, ActivationFunction, CostFunction, layer::BackpropagationResult};

use nalgebra::{DMatrix, DVector};
use rand::thread_rng;
use serde::{Serialize, Deserialize};
use std::any::{Any};

#[derive(Clone, Serialize, Deserialize)]
pub struct FullyConnected {
    pub weights: DMatrix<Float>,
    pub biases: DVector<Float>,
    pub activation_function: ActivationFunction
}

impl FullyConnected {

    pub fn new(previous_layer_size: usize, layer_size: usize, activation_function: ActivationFunction) -> FullyConnected {
        let mut rng = thread_rng();

        let weights = DMatrix::<Float>::zeros(layer_size, previous_layer_size);

        return FullyConnected {
            weights: weights.map(|_| activation_function.initialize_weight(previous_layer_size, &mut rng)),
            biases: DVector::<Float>::zeros(layer_size),
            activation_function
        }
    }

}

#[typetag::serde]
impl Layer for FullyConnected {

    fn trainable(&self) -> bool { true }

    fn feed_forward(&self, input: &DVector<Float>) -> DVector<Float> {
        self.activation(&self.weighted_input(input))
    }

    fn weighted_input(&self, input: &DVector<Float>) -> DVector<Float> {
        &self.weights * input + &self.biases
    }

    fn activation(&self, weighted_input: &DVector<Float>) -> DVector<Float> {
        weighted_input.map(|x| self.activation_function.function(x))
    }

    fn back_propagate(&self, next_error: &mut DVector<Float>, previous_activation: &DVector<Float>, weighted_input: &DVector<Float>, activation: &DVector<Float>, expected_output: &DVector<Float>, cost_function: &CostFunction) -> Box<dyn BackpropagationResult> {
        let error: DVector<Float>;

        if next_error.len() == 0 {
            // If first back propagation layer, initialize error with the output error equation.
            error = cost_function.derivative(activation, expected_output)
                .component_mul(&weighted_input.map(|x| self.activation_function.derivative(x)));
        } else {
            // If not first back propagation layer, multiply the intermediate error value by the activation function derivative of the weighted input.
            error = next_error.component_mul(&weighted_input.map(|x| self.activation_function.derivative(x)));
        }

        let result = FullyConnectedBackpropagationResult {
            delta_weight_gradient: &error * previous_activation.transpose(),
            delta_bias_gradient: error.clone()
        };

        *next_error = self.weights.transpose() * error;

        return Box::new(result);
    }

    fn apply_results(&mut self, results: Vec<Box<dyn BackpropagationResult>>, learning_rate: Float) {
        let count = results.len();

        let mut weight_gradient = DMatrix::zeros(0, 0);
        let mut bias_gradient = DVector::zeros(0);

        let mut first = true;

        for result in results {
            let result: &FullyConnectedBackpropagationResult = match result.as_any().downcast_ref() {
                Some(result) => result,
                None => panic!("Incompatible result type for FullyConnected layer: {:?}", result)
            };

            if first {
                weight_gradient = result.delta_weight_gradient.clone();
                bias_gradient = result.delta_bias_gradient.clone();

                first = false;
            } else {
                weight_gradient += &result.delta_weight_gradient;
                bias_gradient += &result.delta_bias_gradient;
            }
        }

        self.weights -= weight_gradient * learning_rate / count as Float;
        self.biases -= bias_gradient * learning_rate / count as Float;
    }

    fn size(&self) -> usize {
        self.biases.len()
    }

}

#[derive(Debug)]
struct FullyConnectedBackpropagationResult {
    delta_weight_gradient: DMatrix<Float>,
    delta_bias_gradient: DVector<Float>
}

impl BackpropagationResult for FullyConnectedBackpropagationResult {

    fn as_any(&self) -> &dyn Any {
        self
    }
    
}