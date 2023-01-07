use std::any::Any;
use crate::{BackpropagationResult, Float, Layer};
use nalgebra::{DMatrix, DVector};
use rand::{Rng, thread_rng};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// 2-dimensional convolution layer
#[derive(Clone, Serialize, Deserialize)]
pub struct Conv2D {
    pub input_width: usize,
    pub input_height: usize,
    pub filters: Vec<DMatrix<Float>>,
}

impl Conv2D {
    pub fn new(
        filter_count: usize,
        input_width: usize,
        input_height: usize,
        kernel_width: usize,
        kernel_height: usize,
    ) -> Self {
        assert_eq!(
            input_width % kernel_width,
            0,
            "Conv2D kernel width must fit in input width a whole amount of times"
        );
        assert_eq!(
            input_height % kernel_height,
            0,
            "Conv2D kernel height must fit in input height a whole amount of times"
        );

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, (2.0 / (input_width * input_height) as Float).sqrt()).expect("Unable to create normal distribution");

        Self {
            input_width,
            input_height,
            filters: (0..filter_count)
                .into_iter()
                .map(|_| DMatrix::from_fn(kernel_height, kernel_width, |_, _| rng.sample(normal)))
                .collect(),
        }
    }

    pub fn new_square(filter_count: usize, input_size: usize, kernel_size: usize) -> Self {
        Self::new(
            filter_count,
            input_size,
            input_size,
            kernel_size,
            kernel_size,
        )
    }

    fn convolve(input: &DMatrix<Float>, kernel: &DMatrix<Float>) -> DMatrix<Float> {
        let (input_height, input_width) = input.shape();
        let (kernel_height, kernel_width) = kernel.shape();

        let output_width = input_width - kernel_width + 1;
        let output_height = input_height - kernel_height + 1;

        let mut output = DMatrix::zeros(output_height, output_width);

        for y in 0..output_height {
            for x in 0..output_width {
                output[(y, x)] = input
                    .slice((y, x), (kernel_height, kernel_width))
                    .component_mul(kernel)
                    .sum()
            }
        }

        return output
    }

    fn convolve_append(input: &DMatrix<Float>, kernel: &DMatrix<Float>, output: &mut DVector<Float>) {
        let (input_height, input_width) = input.shape();
        let (kernel_height, kernel_width) = kernel.shape();

        let output_width = input_width - kernel_width + 1;
        let output_height = input_height - kernel_height + 1;

        output.resize_vertically_mut(output.nrows() + output_width * output_height, 0 as Float);

        for y in 0..output_height {
            for x in 0..output_width {
                output[x * output_height + y] = input
                    .slice((y, x), (kernel_height, kernel_width))
                    .component_mul(kernel)
                    .sum()
            }
        }
    }

}

#[typetag::serde]
impl Layer for Conv2D {
    fn trainable(&self) -> bool {
        true
    }

    fn feed_forward(&self, input: &DVector<Float>) -> DVector<Float> {
        self.weighted_input(input)
    }

    fn weighted_input(&self, input: &DVector<Float>) -> DVector<Float> {
        let mut output = DVector::<Float>::zeros(0);

        let input = DMatrix::from_column_slice(
            self.input_height,
            self.input_width,
            input.data.as_slice()
        );

        for filter in &self.filters {
            Conv2D::convolve_append(&input, filter, &mut output);
        }

        return output;
    }

    fn activation(&self, weighted_input: &DVector<Float>) -> DVector<Float> {
        weighted_input.clone()
    }

    fn back_propagate(
        &self,
        next_error: &mut DVector<Float>,
        _previous_activation: &DVector<Float>,
        _weighted_input: &DVector<Float>,
    ) -> Box<dyn BackpropagationResult> {
        let input = DMatrix::from_column_slice(
            self.input_height,
            self.input_width,
            _previous_activation.data.as_slice()
        );

        let mut result = Conv2DBackpropagationResult {
            delta_filter_gradients: vec![]
        };

        let mut offset = 0;

        for filter in &self.filters {
            let (filter_height, filter_width) = filter.shape();

            let output_width = self.input_width - filter_width + 1;
            let output_height = self.input_height - filter_height + 1;

            let filter_error_data = &input.data.as_slice()[offset..(offset + output_width * output_height)];
            assert_eq!(filter_error_data.len(), output_width * output_height);

            let mut filter_error = DMatrix::from_column_slice(output_height, output_width, filter_error_data);

            // Flip the error matrix
            for column in 0..(filter_error.ncols() / 2) {
                filter_error.swap_columns(column, filter_error.ncols() - 1 - column);
            }

            for row in 0..(filter_error.nrows() / 2) {
                filter_error.swap_rows(row, filter_error.nrows() - 1 - row);
            }

            let filter_gradient = Conv2D::convolve(&input, &filter_error);

            result.delta_filter_gradients.push(filter_gradient);

            offset += output_width * output_height;
        }

        return Box::new(result);
    }

    fn apply_results(
        &mut self,
        results: Vec<Box<dyn BackpropagationResult>>,
        learning_rate: Float,
    ) {
        let count = results.len();

        let mut filter_gradients: Vec<DMatrix<Float>> = vec![];

        let mut first = true;

        for result in results {
            let result: &Conv2DBackpropagationResult = match result.as_any().downcast_ref()
            {
                Some(result) => result,
                None => panic!(
                    "Incompatible result type for Conv2D layer: {:?}",
                    result
                ),
            };

            if first {
                filter_gradients = result.delta_filter_gradients.clone();

                first = false;
            } else {
                filter_gradients.iter_mut().zip(result.delta_filter_gradients.iter()).for_each(|(g, d)| *g += d);
            }
        }

        self.filters.iter_mut().zip(filter_gradients.iter()).for_each(|(f, g)| *f -= g * learning_rate / count as Float);
    }

    fn size(&self) -> usize {
        self.filters.iter().map(|f| (self.input_width - f.ncols() + 1) * (self.input_height - f.nrows() + 1)).sum()
    }
}

#[derive(Debug)]
struct Conv2DBackpropagationResult {
    delta_filter_gradients: Vec<DMatrix<Float>>
}

impl BackpropagationResult for Conv2DBackpropagationResult {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::layer::Conv2D;
    use crate::{Float, Layer};
    use nalgebra::{DMatrix, DVector, Matrix4, ReshapableStorage};

    fn sample_image() -> Matrix4<Float> {
        Matrix4::new(
            1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        )
    }

    #[test]
    fn feed_forward() {
        let input_image = sample_image();
        let input = DVector::from_vec(input_image.data.as_slice().to_vec());

        let layer = Conv2D::new_square(3, 4, 2);

        println!("{:?}", layer.feed_forward(&input));
    }

    #[test]
    fn back_propagation() {
        let input_image = sample_image();

        let previous_activation = DVector::from_vec(input_image.data.as_slice().to_vec());
        let weighted_input = DVector::from_vec(vec![6.0, 14.0, 8.0, 16.0, 13.0, 2.0, 6.0, 9.0, 11.0]);
        let mut error = DVector::from_vec(vec![1.0; 9]);

        let layer = Conv2D::new_square(1, 4, 2);

        let result = layer.back_propagate(&mut error, &previous_activation, &weighted_input);
        println!("{:?} {:?}", result, error);
        assert_eq!(0, 1);
    }
}
