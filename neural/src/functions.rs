use rand::Rng;
use rand_distr::{Normal};
use rand_distr::num_traits::Pow;

#[derive(bincode::Encode, bincode::Decode)]
pub enum ActivationFunction {
    Input, Sigmoid, ReLU, LeakyReLU(f64), Tanh
}

impl ActivationFunction {
    pub fn function(&self, x: f64) -> f64 {
        match self {
            Self::Input => panic!("Input does not have an activation function"),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::ReLU => x.max(0.0),
            Self::LeakyReLU(alpha) => if x >= 0.0 { x } else { x * alpha },
            Self::Tanh => x.tanh()
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Input => panic!("Input does not have an activation function"),
            Self::Sigmoid => self.function(x) * (1.0 - self.function(x)),
            Self::ReLU => if x >= 0.0 { 1.0 } else { 0.0 },
            Self::LeakyReLU(alpha) => if x >= 0.0 { 1.0 } else { *alpha },
            Self::Tanh => 1.0 - x.tanh().pow(2)
        }
    }

    pub fn initialize_weight(&self, previous_layer_size: usize, rng: &mut impl Rng) -> f64 {
        match self {
            Self::Input => panic!("Input does not have weights"),
            Self::Sigmoid | Self::Tanh => {
                let bound = 1.0 / (previous_layer_size as f64).sqrt();

                return rng.gen_range((-bound)..(bound));
            },
            Self::ReLU | Self::LeakyReLU(_) => {
                let deviation = (2.0 / previous_layer_size as f64).sqrt();
                let normal = Normal::new(0.0, deviation).expect("Couldn't create normal distribution");

                return rng.sample(normal);
            }
        }
    }

    pub fn from(string: &str) -> Option<Self> {
        if string.starts_with("leakyrelu(") && string.ends_with(")") {
            let alpha_start = string.find("(").expect("Couldn't find open");
            let alpha_end = string.find(")").expect("Couldn't find end");

            let alpha = string[alpha_start + 1..alpha_end].parse::<f64>().expect("Couldn't parse alpha value");

            return Some(Self::LeakyReLU(alpha))
        }

        match string.to_lowercase().as_str() {
            "input" => Some(Self::Input),
            "sigmoid" => Some(Self::Sigmoid),
            "relu" => Some(Self::ReLU),
            "tanh" => Some(Self::Tanh),
            _ => None
        }
    }
}
