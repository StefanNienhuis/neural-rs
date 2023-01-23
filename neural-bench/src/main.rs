extern crate core;

use std::io;
use std::path::{PathBuf};
use neural::{CostFunction, Network, layer, ActivationFunction, Float};
use neural_utils::{io::read_idx_file, outputs_from_labels};

const BATCH_SIZE: usize = 10;
const LEARNING_RATE: Float = 0.01;

fn test(network: &Network, test_data: &[(Vec<Float>, u8)]) -> Float {
    let mut result: Float = 0 as Float;

    for (input, label) in test_data {
        let output = network.feed_forward(input.clone());

        let (output, _) = output
            .iter()
            .enumerate()
            .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
            .unwrap();

        if output == usize::from(*label) {
            result += 1.0;
        }

    }

    return result / test_data.len() as Float;
}

fn epochs() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = Network::new(CostFunction::MeanSquaredError);

    network.add_layer(layer::Input::new(28 * 28));
    network.add_layer(layer::FullyConnected::new(28 * 28, 392, ActivationFunction::ReLU));
    network.add_layer(layer::FullyConnected::new(392, 196, ActivationFunction::ReLU));
    network.add_layer(layer::FullyConnected::new(196, 98, ActivationFunction::ReLU));
    network.add_layer(layer::FullyConnected::new(98, 49, ActivationFunction::ReLU));
    network.add_layer(layer::FullyConnected::new(49, 27, ActivationFunction::Sigmoid));

    let training_images = read_idx_file(&PathBuf::from("./data/emnist/letters/train-images"))?;
    let training_labels = read_idx_file(&PathBuf::from("./data/emnist/letters/train-labels"))?;
    let test_images = read_idx_file(&PathBuf::from("./data/emnist/letters/test-images"))?;
    let test_labels = read_idx_file(&PathBuf::from("./data/emnist/letters/test-labels"))?;

    let training_data: Vec<(Vec<Float>, Vec<Float>)> = training_images.items.clone()
        .into_iter()
        .map(|input| {
            input
                .into_iter()
                .map(|x| Float::from(x) / 255.0)
                .collect::<Vec<_>>()
        })
        .zip(outputs_from_labels(&network, training_labels.items.clone()).into_iter())
        .collect();

    let train_test_data: Vec<(Vec<Float>, u8)> = training_images.items
        .into_iter()
        .map(|input| {
            input
                .into_iter()
                .map(|x| Float::from(x) / 255.0)
                .collect::<Vec<_>>()
        })
        .zip(training_labels.items.into_iter().map(|x| x[0]))
        .collect();

    let test_data: Vec<(Vec<Float>, u8)> = test_images.items
        .into_iter()
        .map(|input| {
            input
                .into_iter()
                .map(|x| Float::from(x) / 255.0)
                .collect::<Vec<_>>()
        })
        .zip(test_labels.items.into_iter().map(|x| x[0]))
        .collect();

    let mut writer = csv::Writer::from_writer(io::stdout());

    writer.write_record(&["epochs", "train_accuracy", "test_accuracy"])?;
    writer.flush()?;

    let train_result = test(&network, &train_test_data);
    let test_result = test(&network, &test_data);

    writer.write_record(&["0", format!("{}", train_result).as_str(), format!("{}", test_result).as_str()])?;
    writer.flush()?;

    for epoch in 1..=30 {
        network.stochastic_gradient_descent(training_data.clone(), BATCH_SIZE, LEARNING_RATE);

        let train_result = test(&network, &train_test_data);
        let test_result = test(&network, &test_data);

        writer.write_record(&[format!("{}", epoch), format!("{}", train_result), format!("{}", test_result)])?;
        writer.flush()?;
    }

    Ok(())
}

fn main() {
    epochs().unwrap();
}
