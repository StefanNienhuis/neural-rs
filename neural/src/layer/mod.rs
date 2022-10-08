mod input;
mod fully_connected;

pub use input::Input;
pub use fully_connected::FullyConnected;

use crate::{Float, CostFunction};

use nalgebra::{DMatrix, DVector};

#[typetag::serde(tag = "type")]
pub trait Layer: Sync {
    /// Calculates the activation based on the input
    fn feed_forward(&self, input: &DVector<Float>) -> DVector<Float>;

    /// Calculates the weighted input based on the input.
    fn weighted_input(&self, input: &DVector<Float>) -> DVector<Float>;

    /// Calculates the activation based on the weighted input.
    fn activation(&self, weighted_input: &DVector<Float>) -> DVector<Float>;

    /// A zeroed out weight vector for backpropagation initialization.
    /// Should have the same shape as the weights.
    fn zeroed_weight_gradient(&self) -> DMatrix<Float>;

    /// A zeroed out bias vector for backpropagation initialization.
    /// Should have the same shape as the bias.
    fn zeroed_bias_gradient(&self) -> DVector<Float>;

    /// Calculate the error in the current layer, if it is the output.
    fn output_error(&self, cost_function: &CostFunction, weighted_input: &DVector<Float>, output: &DVector<Float>, expected_output: &DVector<Float>) -> DVector<Float>;

    /// Calculate the error in the previous layer
    fn error(&self, weighted_input: &DVector<Float>, previous_error: &DVector<Float>) -> DVector<Float>;

    /// Apply the weight result gradient from backpropagation to the layer.
    fn apply_weight_gradient(&mut self, gradient: DMatrix<Float>);

    /// Apply the bias result gradient from backpropagation to the layer.
    fn apply_bias_gradient(&mut self, gradient: DVector<Float>);

    /// The size of this layer
    fn size(&self) -> usize;
}


