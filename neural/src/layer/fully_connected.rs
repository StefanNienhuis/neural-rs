use crate::{Layer, Float, ActivationFunction, CostFunction};

use nalgebra::{DMatrix, DVector};
use rand::thread_rng;
use serde::{Serialize, Deserialize};

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

    fn feed_forward(&self, input: &DVector<Float>) -> DVector<Float> {
        self.activation(&self.weighted_input(input))
    }

    fn weighted_input(&self, input: &DVector<Float>) -> DVector<Float> {
        &self.weights * input + &self.biases
    }

    fn activation(&self, weighted_input: &DVector<Float>) -> DVector<Float> {
        weighted_input.map(|x| self.activation_function.function(x))
    }

    fn zeroed_weight_gradient(&self) -> DMatrix<Float> {
        DMatrix::zeros(self.weights.nrows(), self.weights.ncols())
    }

    fn zeroed_bias_gradient(&self) -> DVector<Float> {
        DVector::zeros(self.biases.nrows())
    }

    fn output_error(&self, cost_function: &CostFunction, weighted_input: &DVector<Float>, output: &DVector<Float>, expected_output: &DVector<Float>) -> DVector<Float> {
        cost_function.derivative(output, expected_output)
            .component_mul(&weighted_input.map(|x| self.activation_function.derivative(x)))
    }

    fn error(&self, weighted_input: &DVector<Float>, previous_error: &DVector<Float>) -> DVector<Float> {
        (self.weights.transpose() * previous_error)
            .component_mul(&weighted_input.map(|x| self.activation_function.derivative(x)))
    }

    fn apply_weight_gradient(&mut self, gradient: DMatrix<Float>) {
        self.weights -= gradient;
    }

    fn apply_bias_gradient(&mut self, gradient: DVector<Float>) {
        self.biases -= gradient;
    }

    fn size(&self) -> usize {
        self.biases.len()
    }

}