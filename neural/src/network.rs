use crate::math;

// A 3 layer neural network.
// Until Rust supports array const generics, I can't think of a way to do this in a type safe manner.
#[derive(bincode::Encode, bincode::Decode)]
pub struct Network3<const I: usize, const H0: usize, const O: usize> {
    pub weights: ([[f64; I]; H0], [[f64; H0]; O]),
    pub biases: ([f64; H0], [f64; O])
}

impl<const I: usize, const H0: usize, const O: usize> Network3<I, H0, O> {

    pub fn zeros() -> Self {
        return Self {
            weights: ([[0f64; I]; H0], [[0f64; H0]; O]),
            biases: ([0f64; H0], [0f64; O]),
        }
    }

    pub fn random() -> Self {
        // I hate this solution, but I can't think of a better way.
        let mut network = Self::zeros();

        math::fill_random(&mut network.biases.0);
        math::fill_random(&mut network.biases.1);

        for i in 0..network.weights.0.len() {
            math::fill_random(&mut network.weights.0[i]);
        }

        for i in 0..network.weights.1.len() {
            math::fill_random(&mut network.weights.1[i]);
        }

        return network;
    }

    pub fn feed_forward(&self, input: [f64; I]) -> [f64; O] {
        let h0 = math::layer(input, self.weights.0, self.biases.0);
        let output = math::layer(h0, self.weights.1, self.biases.1);

        return output;
    }

}