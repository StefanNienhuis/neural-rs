use rand::prelude::*;

pub fn sigmoid(x: f64) -> f64 {
    return 1f64 / (1f64 + x.exp());
}

pub fn layer<const I: usize, const O: usize>(input: [f64; I], weights: [[f64; I]; O], biases: [f64; O]) -> [f64; O] {
    let mut activation = [0f64; O];

    for (i, (weight, bias)) in weights.iter().zip(biases.iter()).enumerate() {
        activation[i] = sigmoid(input.iter().enumerate().map(|(j, input)| input * weight[j]).sum::<f64>() + bias);
    }

    return activation;
}

pub fn fill_random<const S: usize>(array: &mut [f64; S]) {
    let mut rng = thread_rng();

    for i in 0..array.len() {
        array[i] = rng.gen::<f64>();
    }
}