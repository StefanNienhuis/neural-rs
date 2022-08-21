pub mod network;
pub mod back_propagation;
pub mod functions;

#[cfg(feature = "high-precision")]
pub type Float = f64;

#[cfg(not(feature = "high-precision"))]
pub type Float = f32;