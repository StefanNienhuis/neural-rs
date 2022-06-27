pub fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}