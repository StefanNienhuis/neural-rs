pub mod network;
pub mod back_propagation;
pub mod activation_function;
pub mod cost_function;
pub mod layer;

pub use self::{network::Network, activation_function::ActivationFunction, cost_function::CostFunction, layer::Layer};

#[cfg(feature = "high-precision")]
pub type Float = f64;

#[cfg(not(feature = "high-precision"))]
pub type Float = f32;