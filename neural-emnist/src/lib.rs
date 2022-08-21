mod utils;

use wasm_bindgen::prelude::*;
use neural::Float;

#[wasm_bindgen(start)]
pub fn main() {
    utils::set_panic_hook();
}

#[wasm_bindgen]
pub struct Network {
    network: neural::network::Network
}

#[wasm_bindgen]
impl Network {

    #[wasm_bindgen(constructor)]
    pub fn new(network: &[u8]) -> Self {
        return Self {
            network: match bincode::decode_from_slice(network, bincode::config::standard()) {
                Err(error) => panic!("Couldn't parse network: {}", error),
                Ok((network, _)) => network
            }
        }
    }

    /// Returns the shape of the network.
    ///
    /// **Return value:** Uint32Array containing the sizes of each layer.
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        return self.network.shape();
    }

    /// Returns the weights of the network.
    ///
    /// **Return value:** Float64Array containing the flattened weights. Formatted as NxL<sub>n</sub>xL<sub>n-1</sub>, starting at the first hidden layer.
    #[wasm_bindgen(getter)]
    pub fn weights(&self) -> Box<[Float]> {
        return self.network.weights.iter().map(|array| array.as_slice().to_vec()).flatten().collect::<Vec<Float>>().into_boxed_slice();
    }

    /// Returns the biases of the network.
    ///
    /// **Return value:** Float64Array containing the flattened biases. Formatted as NxL<sub>n</sub>, starting at the first hidden layer.
    #[wasm_bindgen(getter)]
    pub fn biases(&self) -> Box<[Float]> {
        return self.network.biases.iter().map(|array| array.as_slice().to_vec()).flatten().collect::<Vec<Float>>().into_boxed_slice();
    }

    /// Feeds forward the input through the network and returns the output.
    ///
    /// **Return value:** Float64Array containing the output.
    pub fn feed_forward(&self, input: &[Float]) -> Box<[Float]> {
        return self.network.feed_forward(input.to_vec()).into_boxed_slice();
    }

}