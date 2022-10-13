mod fully_connected;
mod input;
mod pool2d;

pub use fully_connected::FullyConnected;
pub use input::Input;
pub use pool2d::{Pool2D, PoolType};

use crate::Float;
use nalgebra::DVector;
use std::any::Any;
use std::fmt::Debug;

#[typetag::serde(tag = "type")]
pub trait Layer: Sync {
    /// Boolean indicating whether this layer is trainable.
    fn trainable(&self) -> bool;

    /// Calculates the activation based on the input
    fn feed_forward(&self, input: &DVector<Float>) -> DVector<Float> {
        self.activation(&self.weighted_input(input))
    }

    /// Calculates the weighted input based on the input.
    fn weighted_input(&self, input: &DVector<Float>) -> DVector<Float>;

    /// Calculates the activation based on the weighted input.
    fn activation(&self, weighted_input: &DVector<Float>) -> DVector<Float>;

    /// Back propagate the error through this layer.
    /// Next error is the intermediate error value from the previous layer, which will be updated and passed on to the next layer.
    /// If next error is uninitialized (len == 0), the implementation must initialize this according to the cost function formula.
    /// Returns a back propagation result struct (e.g. delta_weight_gradient, delta_bias_gradient).
    fn back_propagate(
        &self,
        next_error: &mut DVector<Float>,
        previous_activation: &DVector<Float>,
        weighted_input: &DVector<Float>,
    ) -> Box<dyn BackpropagationResult>;

    fn apply_results(&mut self, results: Vec<Box<dyn BackpropagationResult>>, learning_rate: Float);

    /// The size of this layer
    fn size(&self) -> usize;
}

pub trait BackpropagationResult: Debug {
    fn as_any(&self) -> &dyn Any;
}

impl BackpropagationResult for () {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
