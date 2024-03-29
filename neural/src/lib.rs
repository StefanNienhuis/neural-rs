extern crate core;

pub mod activation_function;
pub mod back_propagation;
pub mod cost_function;
pub mod layer;
pub mod network;

pub use self::{
    activation_function::ActivationFunction, cost_function::CostFunction,
    layer::BackpropagationResult, layer::Layer, layer::LayerEnum, network::Network,
};

#[cfg(feature = "high-precision")]
pub type Float = f64;

#[cfg(not(feature = "high-precision"))]
pub type Float = f32;
