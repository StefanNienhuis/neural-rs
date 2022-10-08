use crate::{Float, Layer, CostFunction};

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
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

    fn feed_forward(&self, input: &DVector<Float>) -> DVector<Float> {
        input.clone()
    }

    fn weighted_input(&self, input: &DVector<Float>) -> DVector<Float> {
        input.clone()
    }

    fn activation(&self, weighted_input: &DVector<Float>) -> DVector<Float> {
        weighted_input.clone()
    }

    fn zeroed_weight_gradient(&self) -> DMatrix<Float> {
        DMatrix::zeros(0,0)
    }

    fn zeroed_bias_gradient(&self) -> DVector<Float> {
        DVector::zeros(0)
    }

    fn output_error(&self, _cost_function: &CostFunction, _weighted_input: &DVector<Float>, _output: &DVector<Float>, _expected_output: &DVector<Float>) -> DVector<Float> {
        panic!("Output error cannot be calculated on input layer")
    }

    fn error(&self, _weighted_input: &DVector<Float>, _previous_error: &DVector<Float>) -> DVector<Float> {
        panic!("Error cannot be calculated on input layer")
    }

    fn apply_weight_gradient(&mut self, _gradient: DMatrix<Float>) {
        panic!("Cannot apply weight gradient to input layer");
    }

    fn apply_bias_gradient(&mut self, _gradient: DVector<Float>) {
        panic!("Cannot apply bias gradient to input layer");
    }

    fn size(&self) -> usize {
        self.size
    }

}