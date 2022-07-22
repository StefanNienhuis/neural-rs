use crate::{network::{Network}};

use nalgebra::{DMatrix, DVector};
use rand::{seq::SliceRandom};

#[cfg(feature = "threads")]
use crossbeam_utils::{thread};

impl Network {

    /// Train the network using stochastic gradient descent
    ///
    /// *Single threaded*
    pub fn stochastic_gradient_descent(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>, batch_size: usize, learning_rate: f64) {
        let mut training_data: Vec<(DVector<f64>, DVector<f64>)> = training_data.into_iter().map(|(input, output)| (input.into(), output.into())).collect();

        let mut rng = rand::thread_rng();

        training_data.shuffle(&mut rng);

        for batch in training_data.chunks(batch_size) {
            let (weight_gradients, bias_gradients) = self.train_sgd_batch(batch, &self.weights, &self.biases);

            self.weights = self.weights.iter().zip(weight_gradients.into_iter()).map(|(weights, gradient)| weights - (learning_rate / batch.len() as f64) * gradient).collect();
            self.biases = self.biases.iter().zip(bias_gradients.into_iter()).map(|(biases, gradient)| biases - (learning_rate / batch.len() as f64) * gradient).collect();
        }
    }

    #[cfg(feature = "threads")]
    /// Train the network using parallel stochastic gradient descent
    ///
    /// *Multithreaded*
    pub fn parallel_stochastic_gradient_descent(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>, thread_count: usize, batch_size: usize, learning_rate: f64) {
        let mut training_data: Vec<(DVector<f64>, DVector<f64>)> = training_data.into_iter().map(|(input, output)| (input.into(), output.into())).collect();

        assert_eq!(training_data.len() % batch_size, 0, "Training data must be split equally over batches");
        assert_eq!((training_data.len() / batch_size) % thread_count, 0, "Batches must be split equally over threads");

        let mut rng = rand::thread_rng();

        training_data.shuffle(&mut rng);

        let mut new_weights = self.weights.iter().map(|x| DMatrix::<f64>::zeros(x.nrows(), x.ncols())).collect::<Vec<_>>();
        let mut new_biases = self.biases.iter().map(|x| DVector::<f64>::zeros(x.nrows())).collect::<Vec<_>>();

        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);

            let this = &self;

            for (i, batches) in training_data.chunks(training_data.len() / thread_count).enumerate() {
                handles.push(scope.spawn(move |_| {
                    println!("Thread {} spawned with {} batches", i, batches.len());

                    let mut weights = this.weights.clone();
                    let mut biases = this.biases.clone();

                    for batch in batches.chunks(batch_size) {
                        let (weight_gradients, bias_gradients) = this.train_sgd_batch(batch, &weights, &biases);

                        weights = weights.iter().zip(weight_gradients.into_iter()).map(|(weights, gradient)| weights - (learning_rate / batch.len() as f64) * gradient).collect();
                        biases = biases.iter().zip(bias_gradients.into_iter()).map(|(biases, gradient)| biases - (learning_rate / batch.len() as f64) * gradient).collect();
                    }

                    println!("Thread {} done", i);

                    return (weights, biases);
                }));
            }

            for handle in handles {
                let (thread_weights, thread_biases) = handle.join().unwrap();

                new_weights.iter_mut().enumerate().for_each(|(i, x)| *x += &thread_weights[i] / thread_count as f64);
                new_biases.iter_mut().enumerate().for_each(|(i, x)| *x += &thread_biases[i] / thread_count as f64)
            }

        }).unwrap();

        self.weights = new_weights;
        self.biases = new_biases;
    }

    /// Calculate the weight and bias gradients for a specific SGD batch
    fn train_sgd_batch(&self, batch: &[(DVector<f64>, DVector<f64>)], weights: &Vec<DMatrix<f64>>, biases: &Vec<DVector<f64>>) -> (Vec<DMatrix<f64>>, Vec<DVector<f64>>) {
        let mut weight_gradients: Vec<DMatrix<f64>> = weights.iter().map(|weights| DMatrix::zeros(weights.nrows(), weights.ncols())).collect();
        let mut bias_gradients: Vec<DVector<f64>> = biases.iter().map(|biases| DVector::zeros(biases.nrows())).collect();

        // Calculate the weight and bias gradients for a specific training sample
        for (input, expected_output) in batch {
            let (delta_weight_gradients, delta_bias_gradients) = self.back_propagate(input, expected_output, weights, biases);

            weight_gradients = weight_gradients.iter().zip(delta_weight_gradients.iter()).map(|(gradient, delta)| gradient + delta).collect();
            bias_gradients = bias_gradients.iter().zip(delta_bias_gradients.iter()).map(|(gradient, delta)| gradient + delta).collect();
        }

        return (weight_gradients, bias_gradients);
    }

    fn back_propagate(&self, input: &DVector<f64>, expected_output: &DVector<f64>, weights: &Vec<DMatrix<f64>>, biases: &Vec<DVector<f64>>) -> (Vec<DMatrix<f64>>, Vec<DVector<f64>>) {
        let mut weight_gradients: Vec<DMatrix<f64>> = weights.iter().map(|weights| DMatrix::zeros(weights.nrows(), weights.ncols())).collect();
        let mut bias_gradients: Vec<DVector<f64>> = biases.iter().map(|biases| DVector::zeros(biases.nrows())).collect();

        let mut activation = input.clone();
        let mut activations: Vec<DVector<f64>> = vec![input.clone()];

        // Activations before the non-linear function (e.g. sigmoid) - starts with actual input as this isn't weighted
        let mut weighted_inputs: Vec<DVector<f64>> = vec![input.clone()];

        for (m, (weights, biases)) in weights.iter().zip(biases.iter()).enumerate() {
            let weighted_input = weights * activation + biases;

            activation = weighted_input.map(|x| self.layers[m + 1].activation_function.function(x));

            weighted_inputs.push(weighted_input);
            activations.push(activation.clone());
        }

        let output_layer = activations.len() - 1;

        // Calculate the error in the output layer
        let mut error: DVector<f64> = Self::
            cost_derivative(&activations[output_layer], &expected_output)
            .component_mul(&weighted_inputs[output_layer].map(|x| self.layers.last().expect("No output layer").activation_function.derivative(x)));

        weight_gradients[self.layers.len() - 2] = &error * activations[output_layer - 1].transpose();
        bias_gradients[self.layers.len() - 2] = error.clone();

        // Calculate the errors per layer from the first hidden layer up to (but not including) the output layer
        for l in (1..output_layer).rev() {
            error = (weights[l].transpose() * &error).component_mul(&weighted_inputs[l].map(|x| self.layers[l].activation_function.derivative(x)));

            weight_gradients[l - 1] = &error * activations[l - 1].transpose();
            bias_gradients[l - 1] = error.clone();
        }

        return (weight_gradients, bias_gradients);
    }

    /// Returns the vector of partial derivatives of the cost function with respect to the expected output.
    ///
    /// Formula:
    ///
    /// C = ((output - expected_output)^2)/2
    ///
    /// C' = (2(output - expected_output))/2 =  output - expected_output
    fn cost_derivative(output: &DVector<f64>, expected_output: &DVector<f64>) -> DVector<f64> {
        return output - expected_output;
    }

}