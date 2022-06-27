pub fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

pub fn sigmoid_prime(x: f64) -> f64 {
    return sigmoid(x) * (1.0 - sigmoid(x));
}