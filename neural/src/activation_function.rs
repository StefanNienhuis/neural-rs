use crate::Float;

use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    Input,
    Sigmoid,
    ReLU,
    LeakyReLU(Float),
    Tanh,
}

impl ActivationFunction {
    /// Evaluate the activation function for a value.
    pub fn function(&self, x: Float) -> Float {
        match self {
            Self::Input => panic!("Input does not have an activation function"),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::ReLU => x.max(0.0),
            Self::LeakyReLU(alpha) => {
                if x >= 0.0 {
                    x
                } else {
                    x * alpha
                }
            }
            Self::Tanh => x.tanh(),
        }
    }

    /// Evaluate the derivative of the activation function for a value.
    pub fn derivative(&self, x: Float) -> Float {
        match self {
            Self::Input => panic!("Input does not have an activation function"),
            Self::Sigmoid => self.function(x) * (1.0 - self.function(x)),
            Self::ReLU => {
                if x >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LeakyReLU(alpha) => {
                if x >= 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            Self::Tanh => Float::from(1.0) - x.tanh().powi(2),
        }
    }

    /// Initialize a weight based on the activation function.
    pub fn initialize_weight(&self, previous_layer_size: usize, rng: &mut impl Rng) -> Float {
        match self {
            Self::Input => panic!("Input does not have weights"),
            Self::Sigmoid | Self::Tanh => {
                let bound = 1.0 / (previous_layer_size as Float).sqrt();

                return rng.gen_range((-bound)..(bound));
            }
            Self::ReLU | Self::LeakyReLU(_) => {
                let deviation = (2.0 / previous_layer_size as Float).sqrt();
                let normal =
                    Normal::new(0.0, deviation).expect("Couldn't create normal distribution");

                return rng.sample(normal);
            }
        }
    }

    /// Initialize the activation function from a name string.
    pub fn from(string: &str) -> Option<Self> {
        if string.starts_with("leakyrelu(") && string.ends_with(")") {
            let alpha_start = string.find("(").expect("Couldn't find open");
            let alpha_end = string.find(")").expect("Couldn't find end");

            let alpha = string[alpha_start + 1..alpha_end]
                .parse::<Float>()
                .expect("Couldn't parse alpha value");

            return Some(Self::LeakyReLU(alpha));
        }

        match string.to_lowercase().as_str() {
            "input" => Some(Self::Input),
            "sigmoid" => Some(Self::Sigmoid),
            "relu" => Some(Self::ReLU),
            "tanh" => Some(Self::Tanh),
            _ => None,
        }
    }
}
