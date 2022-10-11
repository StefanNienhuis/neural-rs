use crate::{Float, Layer, CostFunction};

use nalgebra::{DVector};
use serde::{Serialize, Deserialize};
use crate::layer::BackpropagationResult;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Input {
    pub size: usize
}

impl Input {

    pub fn new(size: usize) -> Self {
        Self {
            size
        }
    }

}

#[typetag::serde]
impl Layer for Input {

    fn trainable(&self) -> bool { false }

    fn feed_forward(&self, input: &DVector<Float>) -> DVector<Float> {
        input.clone()
    }

    fn weighted_input(&self, input: &DVector<Float>) -> DVector<Float> {
        input.clone()
    }

    fn activation(&self, weighted_input: &DVector<Float>) -> DVector<Float> {
        weighted_input.clone()
    }

    fn back_propagate(&self, _error: &mut DVector<Float>, _previous_weighted_input: &DVector<Float>, _previous_activation: &DVector<Float>, _weighted_input: &DVector<Float>, _expected_output: &DVector<Float>, _cost_function: &CostFunction) -> Box<dyn BackpropagationResult> {
        return Box::new(());
    }

    fn apply_results(&mut self, _results: Vec<Box<dyn BackpropagationResult>>, _learning_rate: Float) {
        panic!("Cannot apply results to input layer");
    }

    fn size(&self) -> usize {
        self.size
    }

}