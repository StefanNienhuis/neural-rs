use crate::Float;

use nalgebra::DVector;

#[derive(bincode::Encode, bincode::Decode)]
pub enum CostFunction {
    MeanAbsoluteError,
    MeanSquaredError,
}

impl CostFunction {
    // Not ever used in backpropagation, but kept here for reference
    pub fn function(
        &self,
        output: &DVector<Float>,
        expected_output: &DVector<Float>,
    ) -> DVector<Float> {
        match self {
            Self::MeanAbsoluteError => (output - expected_output).abs(),
            Self::MeanSquaredError => (output - expected_output).map(|x| x.powi(2)) / 2.0,
        }
    }

    /// Returns the vector of partial derivatives of the cost function with respect to the expected output
    pub fn derivative(
        &self,
        output: &DVector<Float>,
        expected_output: &DVector<Float>,
    ) -> DVector<Float> {
        match self {
            Self::MeanAbsoluteError => output.zip_map(expected_output, |o, e| {
                if e > o {
                    -1.0
                } else if e < o {
                    1.0
                } else {
                    0.0
                }
            }),
            Self::MeanSquaredError => output - expected_output,
        }
    }

    pub fn from(string: &str) -> Option<Self> {
        match string {
            "mae" | "mean-absolute-error" => Some(Self::MeanAbsoluteError),
            "mse" | "mean-squared-error" => Some(Self::MeanSquaredError),
            _ => None,
        }
    }
}
