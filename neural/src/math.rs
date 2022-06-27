pub fn sigmoid(x: f64) -> f64 {
    return 1f64 / (1f64 + x.exp());
}