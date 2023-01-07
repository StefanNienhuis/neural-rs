use crate::{BackpropagationResult, Float, Layer};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// 2-dimensional pooling layer
/// Input is expected to be a column major vector of values.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pool2D {
    pub pool_type: PoolType,
    pub input_width: usize,
    pub input_height: usize,
    pub kernel_width: usize,
    pub kernel_height: usize,
}

impl Pool2D {
    pub fn new(
        pool_type: PoolType,
        input_width: usize,
        input_height: usize,
        kernel_width: usize,
        kernel_height: usize,
    ) -> Self {
        assert_eq!(
            input_width % kernel_width,
            0,
            "Pool2D kernel width must fit in input width a whole amount of times"
        );
        assert_eq!(
            input_height % kernel_height,
            0,
            "Pool2D kernel height must fit in input height a whole amount of times"
        );

        Self {
            pool_type,
            input_width,
            input_height,
            kernel_width,
            kernel_height,
        }
    }

    pub fn new_square(pool_type: PoolType, input_size: usize, kernel_size: usize) -> Self {
        Self::new(pool_type, input_size, input_size, kernel_size, kernel_size)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PoolType {
    MAX,
    AVERAGE,
}

impl FromStr for PoolType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "max" => Ok(Self::MAX),
            "avg" | "average" => Ok(Self::AVERAGE),
            _ => Err(()),
        }
    }
}

#[typetag::serde]
impl Layer for Pool2D {
    fn trainable(&self) -> bool {
        false
    }

    fn feed_forward(&self, input: &DVector<Float>) -> DVector<Float> {
        self.weighted_input(input)
    }

    fn weighted_input(&self, input: &DVector<Float>) -> DVector<Float> {
        assert_eq!(
            input.len(),
            self.input_width * self.input_height,
            "Incorrect input length {}. Should be {} * {} = {}",
            input.len(),
            self.input_width,
            self.input_height,
            self.input_width * self.input_height
        );

        let input = DMatrix::from_vec(
            self.input_height,
            self.input_width,
            input.data.as_vec().clone(),
        );

        let mut output = DVector::<Float>::zeros(self.size());

        for x in 0..(self.input_width / self.kernel_width) {
            for y in 0..(self.input_height / self.kernel_height) {
                let pool = input
                    .slice(
                        (y * self.kernel_height, x * self.kernel_width),
                        (self.kernel_height, self.kernel_width),
                    )
                    .into_owned();

                output[x * (self.input_width / self.kernel_width) + y] = match self.pool_type {
                    PoolType::MAX => pool.max(),
                    PoolType::AVERAGE => {
                        pool.sum() / (self.kernel_width * self.kernel_height) as f64
                    }
                };
            }
        }

        return output;
    }

    fn activation(&self, weighted_input: &DVector<Float>) -> DVector<Float> {
        weighted_input.clone()
    }

    fn back_propagate(
        &self,
        next_error: &mut DVector<Float>,
        previous_activation: &DVector<Float>,
        weighted_input: &DVector<Float>,
    ) -> Box<dyn BackpropagationResult> {
        // Activation is linear, so no Hadamard product needed
        let mut next_error_matrix = DMatrix::<Float>::zeros(self.input_height, self.input_width);

        for x in 0..self.input_width {
            let px = x / self.kernel_width;

            for y in 0..self.input_height {
                let py = y / self.kernel_height;

                match self.pool_type {
                    PoolType::MAX => {
                        next_error_matrix[x * self.input_width + y] = if weighted_input
                            [px * (self.input_width / self.kernel_width) + py]
                            == previous_activation[x * self.input_width + y]
                        {
                            next_error[px * (self.input_width / self.kernel_width) + py]
                        } else {
                            0.0
                        }
                    }
                    PoolType::AVERAGE => {
                        next_error_matrix[x * self.input_width + y] =
                            next_error[px * (self.input_width / self.kernel_width) + py] / 4.0
                    }
                }
            }
        }

        *next_error = DVector::from_column_slice(next_error_matrix.data.as_slice());

        return Box::new(());
    }

    fn apply_results(
        &mut self,
        _results: Vec<Box<dyn BackpropagationResult>>,
        _learning_rate: Float,
    ) {
        panic!("Cannot apply results to untrainable layer.")
    }

    fn size(&self) -> usize {
        (self.input_width / self.kernel_width) * (self.input_height / self.kernel_height)
    }
}

#[cfg(test)]
mod tests {
    use super::{Float, Layer, Pool2D, PoolType};
    use nalgebra::{DVector, Matrix4};

    fn sample_image() -> Matrix4<Float> {
        Matrix4::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        )
    }

    #[test]
    fn max_feed_forward() {
        let input_image = sample_image();
        let input = DVector::from_vec(input_image.data.as_slice().to_vec());

        let layer = Pool2D::new_square(PoolType::MAX, 4, 2);

        assert_eq!(
            layer.feed_forward(&input).data.as_vec().clone(),
            vec![6.0, 14.0, 8.0, 16.0]
        )
    }

    #[test]
    fn average_feed_forward() {
        let input_image = sample_image();
        let input = DVector::from_vec(input_image.data.as_slice().to_vec());

        let layer = Pool2D::new_square(PoolType::AVERAGE, 4, 2);

        assert_eq!(
            layer.feed_forward(&input).data.as_vec().clone(),
            vec![3.5, 11.5, 5.5, 13.5]
        )
    }

    #[test]
    fn max_back_propagation() {
        let input_image = sample_image();

        let previous_activation = DVector::from_vec(input_image.data.as_slice().to_vec());
        let weighted_input = DVector::from_vec(vec![6.0, 14.0, 8.0, 16.0]);
        let mut error = DVector::from_vec(vec![4.0; 4]);

        let layer = Pool2D::new_square(PoolType::MAX, 4, 2);

        layer.back_propagate(&mut error, &previous_activation, &weighted_input);

        assert_eq!(
            error.data.as_vec().clone(),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 4.0]
        )
    }

    #[test]
    fn average_back_propagation() {
        let input_image = sample_image();

        let previous_activation = DVector::from_vec(input_image.data.as_slice().to_vec());
        let weighted_input = DVector::from_vec(vec![3.5, 11.5, 5.5, 13.5]);
        let mut error = DVector::from_vec(vec![4.0, 4.0, 4.0, 4.0]);

        let layer = Pool2D::new_square(PoolType::AVERAGE, 4, 2);

        layer.back_propagate(&mut error, &previous_activation, &weighted_input);

        assert_eq!(error.data.as_vec().clone(), vec![1.0; 16])
    }
}
