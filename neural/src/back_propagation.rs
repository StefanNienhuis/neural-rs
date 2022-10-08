use crate::{Network, Float};

use nalgebra::{DMatrix, DVector};
use rand::{seq::SliceRandom};

// #[cfg(feature = "threads")]
// use crossbeam_utils::{thread};

impl Network {

    /// Train the network using stochastic gradient descent
    ///
    /// *Single threaded*
    pub fn stochastic_gradient_descent(&mut self, training_data: Vec<(Vec<Float>, Vec<Float>)>, batch_size: usize, learning_rate: Float) {
        let mut training_data: Vec<(DVector<Float>, DVector<Float>)> = training_data.into_iter().map(|(input, output)| (input.into(), output.into())).collect();

        let mut rng = rand::thread_rng();

        training_data.shuffle(&mut rng);

        for batch in training_data.chunks(batch_size) {
            let (weight_gradients, bias_gradients) = self.train_sgd_batch(batch);

            for i in 1..self.layers.len() {
                self.layers[i].apply_weight_gradient((learning_rate / batch.len() as Float) * &weight_gradients[i - 1]);
                self.layers[i].apply_bias_gradient((learning_rate / batch.len() as Float) * &bias_gradients[i - 1])
            }
        }
    }

    #[cfg(feature = "threads")]
    /// Train the network using parallel stochastic gradient descent
    ///
    /// *Multithreaded*
    pub fn parallel_stochastic_gradient_descent(&mut self, _training_data: Vec<(Vec<Float>, Vec<Float>)>, _thread_count: usize, _batch_size: usize, _learning_rate: Float) {
        todo!()
        // let mut training_data: Vec<(DVector<Float>, DVector<Float>)> = training_data.into_iter().map(|(input, output)| (input.into(), output.into())).collect();
        //
        // assert_eq!(training_data.len() % batch_size, 0, "Training data must be split equally over batches");
        // assert_eq!((training_data.len() / batch_size) % thread_count, 0, "Batches must be split equally over threads");
        //
        // let mut rng = rand::thread_rng();
        //
        // training_data.shuffle(&mut rng);
        //
        // let mut weight_gradients = vec![];
        // let mut bias_gradients = vec![];
        //
        // let mut first = true;
        //
        // thread::scope(|scope| {
        //     let mut handles = Vec::with_capacity(thread_count);
        //
        //     let this = &self;
        //
        //     for (i, batches) in training_data.chunks(training_data.len() / thread_count).enumerate() {
        //         handles.push(scope.spawn(move |_| {
        //             println!("Thread {} spawned with {} batches", i, batches.len());
        //
        //             let mut thread_weight_gradients = vec![];
        //             let mut thread_bias_gradients = vec![];
        //
        //             let mut first = true;
        //
        //             for batch in batches.chunks(batch_size) {
        //                 let (weight_gradients, bias_gradients) = this.train_sgd_batch(batch);
        //
        //                 let weight_gradients = weight_gradients.into_iter().map(|gradient| (learning_rate / batch.len() as Float) * gradient).collect();
        //                 let bias_gradients = bias_gradients.into_iter().map(|gradient| (learning_rate / batch.len() as Float) * gradient).collect();
        //
        //                 if first {
        //                     thread_weight_gradients = weight_gradients;
        //                     thread_bias_gradients = bias_gradients;
        //
        //                     first = false;
        //                 } else {
        //                     thread_weight_gradients = thread_weight_gradients.iter().zip(weight_gradients.iter()).map(|(a, b)| a + b).collect();
        //                     thread_bias_gradients = thread_bias_gradients.iter().zip(bias_gradients.iter()).map(|(a, b)| a + b).collect();
        //                 }
        //             }
        //
        //             println!("Thread {} done", i);
        //
        //             return (thread_weight_gradients, thread_bias_gradients);
        //         }));
        //     }
        //
        //     for handle in handles {
        //         let (thread_weight_gradients, thread_bias_gradients) = handle.join().unwrap();
        //
        //         let thread_weight_gradients = thread_weight_gradients.iter().map(|gradient| gradient / thread_count as Float).collect();
        //         let thread_bias_gradients = thread_bias_gradients.iter().map(|gradient| gradient / thread_count as Float).collect();
        //
        //         if first {
        //             weight_gradients = thread_weight_gradients;
        //             bias_gradients = thread_bias_gradients;
        //
        //             first = false;
        //         } else {
        //             weight_gradients = weight_gradients.iter().zip(thread_weight_gradients.iter()).map(|(a, b)| a + b).collect();
        //             bias_gradients = bias_gradients.iter().zip(thread_bias_gradients.iter()).map(|(a, b)| a + b).collect();
        //         }
        //     }
        //
        // }).unwrap();
        //
        // for i in 1..self.layers.len() {
        //     self.layers[i].apply_weight_gradient(weight_gradients[i - 1].clone());
        //     self.layers[i].apply_bias_gradient(bias_gradients[i - 1].clone());
        // }
    }

    /// Calculate the weight and bias gradients for a specific SGD batch
    fn train_sgd_batch(&self, batch: &[(DVector<Float>, DVector<Float>)]) -> (Vec<DMatrix<Float>>, Vec<DVector<Float>>) {
        let mut weight_gradients: Vec<DMatrix<Float>> = vec![];
        let mut bias_gradients: Vec<DVector<Float>> = vec![];

        let mut first = true;

        // Calculate the weight and bias gradients for a specific training sample
        for (input, expected_output) in batch {
            let (delta_weight_gradients, delta_bias_gradients) = self.back_propagate(input, expected_output);

            // On first run, gradient = 0 + delta = delta
            if first {
                weight_gradients = delta_weight_gradients;
                bias_gradients = delta_bias_gradients;

                first = false;
            } else {
                weight_gradients = weight_gradients.iter().zip(delta_weight_gradients.iter()).map(|(gradient, delta)| gradient + delta).collect();
                bias_gradients = bias_gradients.iter().zip(delta_bias_gradients.iter()).map(|(gradient, delta)| gradient + delta).collect();
            }
        }

        return (weight_gradients, bias_gradients);
    }

    fn back_propagate(&self, input: &DVector<Float>, expected_output: &DVector<Float>) -> (Vec<DMatrix<Float>>, Vec<DVector<Float>>) {
        // Gradients do not have an entry for the input layer
        let mut weight_gradients: Vec<DMatrix<Float>> = self.layers.iter().skip(1).map(|x| x.zeroed_weight_gradient()).collect();
        let mut bias_gradients: Vec<DVector<Float>> = self.layers.iter().skip(1).map(|x| x.zeroed_bias_gradient()).collect();

        let mut activations: Vec<DVector<Float>> = vec![input.clone()];

        // Activations before the activation function - starts with actual input as this isn't weighted
        let mut weighted_inputs: Vec<DVector<Float>> = vec![input.clone()];

        for layer in self.layers.iter().skip(1) {
            let weighted_input = layer.weighted_input(&activations.last().expect("No activations"));

            activations.push(layer.activation(&weighted_input));
            weighted_inputs.push(weighted_input);
        }

        let output_layer = self.layers.len() - 1;

        // Calculate the error in the output layer
        let mut error: DVector<Float> =
            self.layers.last().expect("No layers")
                .output_error(
                    &self.cost_function,
                    weighted_inputs.last().expect("No output"),
                    activations.last().expect("No output"),
                    expected_output);

        weight_gradients[output_layer - 1] = &error * activations[output_layer - 1].transpose();
        bias_gradients[output_layer - 1] = error.clone();

        // Calculate the errors per layer from the last hidden layer to the first
        for i in (1..output_layer).rev() {
            error = self.layers[i + 1].error(&weighted_inputs[i], &error);

            weight_gradients[i - 1] = &error * activations[i - 1].transpose();
            bias_gradients[i - 1] = error.clone();
        }

        return (weight_gradients, bias_gradients);
    }

}