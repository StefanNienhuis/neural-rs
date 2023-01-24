use crate::{Float, Network, Layer};

use crate::layer::BackpropagationResult;
use nalgebra::DVector;
use rand::seq::SliceRandom;

// #[cfg(feature = "threads")]
// use crossbeam_utils::{thread};

impl Network {
    /// Train the network using stochastic gradient descent
    ///
    /// *Single threaded*
    pub fn stochastic_gradient_descent(
        &mut self,
        training_data: Vec<(Vec<Float>, Vec<Float>)>,
        batch_size: usize,
        learning_rate: Float,
    ) {
        let mut training_data: Vec<(DVector<Float>, DVector<Float>)> = training_data
            .into_iter()
            .map(|(input, output)| (input.into(), output.into()))
            .collect();

        let mut rng = rand::thread_rng();

        training_data.shuffle(&mut rng);

        for batch in training_data.chunks(batch_size) {
            let results = self.train_sgd_batch(batch);

            self.layers
                .iter_mut()
                .filter(|l| l.trainable())
                .zip(results.into_iter())
                .for_each(|(l, r)| l.apply_results(r, learning_rate));
        }
    }

    #[cfg(feature = "threads")]
    /// Train the network using parallel stochastic gradient descent
    ///
    /// *Multithreaded*
    pub fn parallel_stochastic_gradient_descent(
        &mut self,
        _training_data: Vec<(Vec<Float>, Vec<Float>)>,
        _thread_count: usize,
        _batch_size: usize,
        _learning_rate: Float,
    ) {
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
    fn train_sgd_batch(
        &self,
        batch: &[(DVector<Float>, DVector<Float>)],
    ) -> Vec<Vec<Box<dyn BackpropagationResult>>> {
        // TODO: Somehow allow addition of dyn BackpropagationResult to remove nested Vec need.
        let mut results: Vec<Vec<Box<dyn BackpropagationResult>>> = vec![];

        let mut first = true;

        // Calculate the weight and bias gradients for a specific training sample
        for (input, expected_output) in batch {
            let batch_results = self.back_propagate(input, expected_output);

            // On first run, gradient = 0 + delta = delta
            if first {
                results = batch_results.into_iter().map(|x| vec![x]).collect();

                first = false;
            } else {
                results
                    .iter_mut()
                    .zip(batch_results)
                    .for_each(|(results, new)| results.push(new));
            }
        }

        return results;
    }

    fn back_propagate(
        &self,
        input: &DVector<Float>,
        expected_output: &DVector<Float>,
    ) -> Vec<Box<dyn BackpropagationResult>> {
        let mut results: Vec<Box<dyn BackpropagationResult>> = vec![];

        let mut activations: Vec<DVector<Float>> = vec![input.clone()];
        let mut weighted_inputs: Vec<DVector<Float>> = vec![input.clone()];

        for layer in self.layers.iter().skip(1) {
            let weighted_input = layer.weighted_input(&activations.last().expect("No activations"));

            activations.push(layer.activation(&weighted_input));
            weighted_inputs.push(weighted_input);
        }

        // Intermediate error value passed between layers
        let mut next_error: DVector<Float> = self
            .cost_function
            .derivative(activations.last().expect("No activations"), expected_output);

        // Calculate the errors per layer from the last hidden layer to the first
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let result = layer.back_propagate(
                &mut next_error,
                if i == 0 {
                    &activations[0]
                } else {
                    &activations[i - 1]
                },
                &weighted_inputs[i],
            );

            if layer.trainable() {
                results.insert(0, result);
            }
        }

        return results;
    }
}
