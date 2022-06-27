use crate::{network::Network, math};

use nalgebra::{DMatrix, DVector};
use rand::{seq::SliceRandom};

// todo: document math
impl Network {

    pub fn stochastic_gradient_descent<F>(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>, batch_size: usize, learning_rate: f64, progress: F) where F: Fn(usize) {
        let mut training_data: Vec<(DVector<f64>, DVector<f64>)> = training_data.iter().map(|(input, output)| (input.clone().into(), output.clone().into())).collect();

        let mut rng = rand::thread_rng();

        training_data.shuffle(&mut rng);

        let batches: Vec<&[(DVector<f64>, DVector<f64>)]> = training_data.chunks(batch_size).collect();

        for (i, batch) in batches.iter().enumerate() {
            self.train_sgd_batch(batch, learning_rate);
            progress(i + 1);
        }
    }

    fn train_sgd_batch(&mut self, batch: &[(DVector<f64>, DVector<f64>)], learning_rate: f64) {
        let mut weight_gradients: Vec<DMatrix<f64>> = self.weights.iter().map(|weights| DMatrix::zeros(weights.nrows(), weights.ncols())).collect();
        let mut bias_gradients: Vec<DVector<f64>> = self.biases.iter().map(|biases| DVector::zeros(biases.nrows())).collect();

        for (input, expected_output) in batch {
            let (delta_weight_gradients, delta_bias_gradients) = self.back_propagate(input.clone(), expected_output.clone());

            weight_gradients = weight_gradients.iter().zip(delta_weight_gradients.iter()).map(|(gradient, delta)| gradient + delta).collect();
            bias_gradients = bias_gradients.iter().zip(delta_bias_gradients.iter()).map(|(gradient, delta)| gradient + delta).collect();
        }

        self.weights = self.weights.iter().zip(weight_gradients.iter()).map(|(weights, gradient)| weights.clone() - (learning_rate / batch.len() as f64) * gradient).collect();
        self.biases = self.biases.iter().zip(bias_gradients.iter()).map(|(biases, gradient)| biases.clone() - (learning_rate / batch.len() as f64) * gradient).collect();
    }

    fn back_propagate(&self, input: DVector<f64>, expected_output: DVector<f64>) -> (Vec<DMatrix<f64>>, Vec<DVector<f64>>) {
        let mut weight_gradients: Vec<DMatrix<f64>> = self.weights.iter().map(|weights| DMatrix::zeros(weights.nrows(), weights.ncols())).collect();
        let mut bias_gradients: Vec<DVector<f64>> = self.biases.iter().map(|biases| DVector::zeros(biases.nrows())).collect();

        let mut activation = input.clone();
        let mut activations: Vec<DVector<f64>> = vec![input.clone()];

        // Activations before the non-linear function (e.g. sigmoid) - starts with actual input as this isn't weighted
        let mut weighted_inputs: Vec<DVector<f64>> = vec![input.clone()];

        for (weights, biases) in self.weights.iter().zip(self.biases.iter()) {
            let weighted_input = weights * activation + biases;

            activation = weighted_input.map(math::sigmoid);

            weighted_inputs.push(weighted_input);
            activations.push(activation.clone());
        }

        let output_layer = activations.len() - 1;

        let mut error: DVector<f64> = Self::
            cost_derivative(&activations[output_layer], &expected_output)
            .component_mul(&weighted_inputs[output_layer].map(math::sigmoid_prime));

        weight_gradients[self.shape.len() - 2] = error.clone() * activations[output_layer - 1].transpose();
        bias_gradients[self.shape.len() - 2] = error.clone();

        // Calculate the errors per layer from the first hidden layer up to (but not including) the output layer
        for l in (1..output_layer).rev() {
            error = (self.weights[l].transpose() * &error).component_mul(&weighted_inputs[l].map(math::sigmoid_prime));

            weight_gradients[l - 1] = &error * activations[l - 1].transpose();
            bias_gradients[l - 1] = error.clone();
        }

        return (weight_gradients, bias_gradients);
    }

    /// Returns the vector of partial derivatives of the cost function with respect to the expected output.
    ///
    /// Formula:
    ///
    /// C = ((expected_output - output)^2)/2
    ///
    /// C' = (2(expected_output- output))/2 = expected_output - output
    fn cost_derivative(output: &DVector<f64>, expected_output: &DVector<f64>) -> DVector<f64> {
        return expected_output - output;
    }

}