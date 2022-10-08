use crate::{CostFunction, Layer, Float};

use nalgebra::{DVector};

#[derive(bincode::Encode, bincode::Decode)]
pub struct Network {
    /// The layers in the network.
    #[bincode(with_serde)]
    pub layers: Vec<Box<dyn Layer>>,

    /// The cost function for the network.
    pub cost_function: CostFunction
}

impl Network {

    pub fn new(cost_function: CostFunction) -> Self {
        return Self {
            layers: vec![],
            cost_function
        }
    }

    pub fn add_layer<L: Layer + Clone + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn shape(&self) -> Vec<usize> {
        return self.layers.iter().map(|l| l.size()).collect();
    }

    pub fn feed_forward(&self, input: Vec<Float>) -> Vec<Float> {
        let mut activation = DVector::from_vec(input) as DVector<Float>;

        for layer in &self.layers {
            activation = layer.feed_forward(&activation);
        }

        return activation.data.into();
    }

}