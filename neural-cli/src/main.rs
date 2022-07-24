mod io;
mod idx;

use bincode;
use std::path::PathBuf;
use std::{process};
use clap::{Parser, Subcommand, ArgAction};
use rand::{seq::SliceRandom};
use neural::{network::Network, functions::*};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,

    #[clap(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new neural network
    Create {
        /// The neural network file
        network: PathBuf,

        /// A layer that should be added to the network
        ///
        /// Specified with [activation_function]:[size].
        /// Must start with input:[size]
        #[clap(short, long="layer", action(ArgAction::Append), min_values(2), required(true))]
        layers: Vec<String>,

        /// The cost function
        #[clap(short, long)]
        cost_function: String
    },

    /// Train a neural network with the provided data set
    Train {
        /// The neural network file
        network: PathBuf,

        /// The input IDX dataset file
        inputs: PathBuf,

        /// The label IDX dataset file
        labels: PathBuf,

        /// The learning rate
        #[clap(short='r', long)]
        learning_rate: f64,

        /// The thread count
        #[clap(short='p', long, default_value = "1")]
        thread_count: usize,

        /// The batch size
        #[clap(short='s', long, default_value = "10")]
        batch_size: usize,

        /// The number of batches to train, defaulting to all batches
        #[clap(short='c', long)]
        batch_count: Option<usize>,

        /// The amount of times it should train
        #[clap(short, long, default_value = "1")]
        epochs: usize,

        /// If provided, tests the network with these inputs at every epoch
        #[clap(long, requires("test-labels"))]
        test_inputs: Option<PathBuf>,

        /// If provided, tests the network with these labels at every epoch
        #[clap(long, requires("test-inputs"))]
        test_labels: Option<PathBuf>
    },

    /// Test a neural network with the provided data set
    Test {
        /// The neural network file
        network: PathBuf,

        /// The input IDX dataset file
        inputs: PathBuf,

        /// The label IDX dataset file
        labels: PathBuf,

        /// The amount of samples to test
        #[clap(short, long)]
        count: Option<usize>
    },

    /// Evaluate a neural network with the provided input
    Evaluate {
        /// The neural network file
        network: PathBuf,

        /// The input file
        input: PathBuf,

        // The output file
        #[clap(short, long, value_name = "FILE")]
        output: Option<PathBuf>
    }

}

fn main() {
    ctrlc::set_handler(|| {
        process::exit(0);
    }).expect("Error while settings Ctrl-C handler");

    let cli = Cli::parse();

    match &cli.command {
        Commands::Create { network, layers, cost_function } => create(network, layers, cost_function),
        Commands::Train { network, inputs, labels, learning_rate, thread_count, batch_size, batch_count, epochs, test_inputs, test_labels } => train(network, inputs, labels, learning_rate, thread_count, batch_size, batch_count, epochs, test_inputs, test_labels, cli.verbose),
        Commands::Test { network, inputs, labels, count } => test(network, inputs, labels, count, cli.verbose),
        Commands::Evaluate { network, input, output } => evaluate(network, input, output)
    }
}

fn create(network_path: &PathBuf, layers: &[String], cost_function: &String) {
    let cost_function = CostFunction::from(cost_function.as_str()).unwrap_or_else(|| panic!("Invalid cost function: {}", cost_function));

    let mut network = Network::new(cost_function);

    for layer in layers {
        let mut split = layer.split(":");
        let activation_function_string = split.next().expect("Missing activation function");
        let size_string = split.next().expect("Missing layer size");

        let activation_function = ActivationFunction::from(activation_function_string).expect("Invalid activation function");
        let size = size_string.parse::<usize>().unwrap_or_else(|error| panic!("Invalid layer size: {}", error));

        network.add_layer(size, activation_function);
    }

    let encoded: Vec<u8> = bincode::encode_to_vec(network, bincode::config::standard()).unwrap();

    match io::write_file(&encoded, network_path, true, Some("nnet")) {
        Err(error) => {
            println!("Couldn't write to {}: {}", network_path.display(), error);
            return;
        },
        Ok(_) => println!("Created a new neural network at {}", network_path.display())
    };
}

fn train(network_path: &PathBuf, inputs_path: &PathBuf, labels_path: &PathBuf,
         learning_rate: &f64, thread_count: &usize, batch_size: &usize, batch_count: &Option<usize>, epochs: &usize,
         test_inputs: &Option<PathBuf>, test_labels: &Option<PathBuf>, verbose: bool) {

    let mut network = match io::read_network_file(network_path) {
        Err(error) => {
            println!("Error while reading network: {}", error);
            return;
        }
        Ok(network) => network
    };

    let inputs: Vec<Vec<_>> = match io::read_idx_file(inputs_path) {
        Err(error) => {
            println!("Error while reading inputs: {}", error);
            return;
        }
        Ok(idx) => idx.items
    };

    let labels = match io::read_idx_file(labels_path) {
        Err(error) => {
            println!("Error while reading labels: {}", error);
            return;
        }
        Ok(idx) => idx.items
    };

    let test_inputs: Option<Vec<Vec<_>>> = match test_inputs {
        Some(test_inputs) => match io::read_idx_file(test_inputs) {
            Err(error) => {
                println!("Error while reading test inputs: {}", error);
                return;
            }
            Ok(idx) => Some(idx.items)
        },
        None => None
    };

    let test_labels: Option<Vec<Vec<_>>> = match test_labels {
        Some(test_labels) => match io::read_idx_file(test_labels) {
            Err(error) => {
                println!("Error while reading test labels: {}", error);
                return;
            }
            Ok(idx) => Some(idx.items)
        }
        None => None
    };

    let inputs_len = inputs.len();

    let training_data: Vec<(Vec<f64>, Vec<f64>)> =
        inputs.into_iter()
            .map(|input| input.into_iter().map(|x| f64::from(x) / 255.0).collect::<Vec<_>>())
            // If expected output is a Vec with a single item, it is seen as an index. If it has multiple items, it is seen as an output.
            .zip(outputs_from_labels(&network, labels).into_iter())
            .collect();

    let mut test_data = match (test_inputs, test_labels) {
        (Some(test_inputs), Some(test_labels)) => Some(test_inputs.into_iter().zip(test_labels.into_iter()).collect::<Vec<(Vec<u8>, Vec<u8>)>>()),
        _ => None
    };

    let mut rng = rand::thread_rng();

    println!();

    for i in 0..*epochs {
        let mut training_data: Vec<(Vec<f64>, Vec<f64>)> = training_data.clone();

        training_data.shuffle(&mut rng);

        training_data = training_data.into_iter().take(
            match batch_count {
                Some(batch_count) => batch_size * batch_count,
                None => inputs_len
            }
        ).collect();

        println!(
            "Starting epoch {} training with {} batches of size {} and {} total samples",
            i + 1,
            match batch_count {
                Some(batch_count) => batch_count.to_string(),
                None => "all".to_string()
            },
            batch_size,
            training_data.len()
        );

        if thread_count > &1 {
            network.parallel_stochastic_gradient_descent(training_data, *thread_count, *batch_size, *learning_rate);
        } else {
            network.stochastic_gradient_descent(training_data, *batch_size, *learning_rate);
        }

        println!("Finished training for epoch {}.", i + 1);

        if let Some(test_data) = &mut test_data {
            test_data.shuffle(&mut rng);
            let accuracy = test_only(&network, test_data, verbose);

            println!("Accuracy: {:.2}%", accuracy * 100.0);
        }

        println!();
    }

    let encoded: Vec<u8> = bincode::encode_to_vec(network, bincode::config::standard()).unwrap();

    match io::write_file(&encoded, network_path, false, Some("nnet")) {
        Err(error) => {
            println!("Error while saving network to {}: {}", network_path.display(), error);
            return;
        },
        Ok(_) => println!("Finished training. Saved the neural network to {}", network_path.display())
    };

}

fn test(network_path: &PathBuf, inputs_path: &PathBuf, labels_path: &PathBuf, count: &Option<usize>, verbose: bool) {
    let network = match io::read_network_file(network_path) {
        Err(error) => {
            println!("Error while reading network: {}", error);
            return;
        }
        Ok(network) => network
    };

    let inputs: Vec<Vec<u8>> = match io::read_idx_file(inputs_path) {
        Err(error) => {
            println!("Error while reading inputs: {}", error);
            return;
        }
        Ok(idx) => idx.items
    };

    let labels: Vec<Vec<u8>> = match io::read_idx_file(labels_path) {
        Err(error) => {
            println!("Error while reading labels: {}", error);
            return;
        }
        Ok(idx) => idx.items
    };

    let count = count.unwrap_or(inputs.len());

    let mut test_data: Vec<(Vec<u8>, Vec<u8>)> = inputs.iter().cloned().zip(labels.iter().cloned()).collect();
    let mut rng = rand::thread_rng();

    test_data.shuffle(&mut rng);

    let test_batch: Vec<(Vec<u8>, Vec<u8>)> = test_data.iter().take(count).cloned().collect();

    let accuracy = test_only(&network, &test_batch, verbose);

    println!("Accuracy: {:.2}%", accuracy * 100.0);
}

fn test_only(network: &Network, test_batch: &[(Vec<u8>, Vec<u8>)], verbose: bool) -> f64 {
    let mut accuracy = 0.0;

    for (input, label) in test_batch {
        let pixels: Vec<f64> = input.iter().map(|x| f64::from(*x) / 255.0).collect();

        let result = network.feed_forward(pixels);

        if label.len() == 1 {
            let label = label[0];
            let (result, probability) = result.iter().enumerate().max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap()).unwrap();

            if result != usize::from(label) {
                if verbose {
                    println!("Wrong: {} = {} @ {:.2}%", label, result, probability * 100.0);
                }
            } else {
                if verbose {
                    println!("Correct: {} = {} @ {:.2}%", label, result, probability * 100.0);
                }

                accuracy += 1.0;
            }
        } else {
            let expected_output = outputs_from_labels(network, vec![label.clone()]).into_iter().next().unwrap();

            let output_len = expected_output.len();

            accuracy += result.into_iter().zip(expected_output.into_iter())
                                          .map(|(r, e)| (r - e).abs()) // Calculate the absolute error
                                          .fold(1.0, |a, x| a - (x / output_len as f64)); // Calculate the accuracy with the average error
        }


    }

    return accuracy / test_batch.len() as f64;
}

fn evaluate(network_path: &PathBuf, input_path: &PathBuf, output_path: &Option<PathBuf>) {
    let network = match io::read_network_file(network_path) {
        Err(error) => {
            println!("{}", error);
            return;
        }
        Ok(network) => network
    };

    let input_data: Vec<f64> = match io::read_file(input_path) {
        Err(error) => {
            println!("Error while reading input: {}", error);
            return;
        }
        Ok(network) => network.iter().map(|x| f64::from(*x) / 255.0).collect()
    };

    if input_data.len() != *network.shape().first().unwrap_or(&0) {
        println!("Incorrect input data length ({}) should be {}", input_data.len(), network.shape().first().unwrap_or(&0));
        return;
    }

    let output = network.feed_forward(input_data);

    if let Some(output_path) = output_path {
        match io::write_file(&output.into_iter().map(|x| (x * 255.0) as u8).collect(), output_path, true, None) {
            Err(error) => println!("Error while saving output: {}", error),
            Ok(_) => println!("Saved output to {}", output_path.display())
        }
    } else {
        for (i, p) in output.iter().enumerate() {
            println!("{:>2}: {:.2}%", i + 1, p * 100.0);
        }
    }
}

fn outputs_from_labels(network: &Network, labels: Vec<Vec<u8>>) -> Vec<Vec<f64>> {
    let labels_len = labels[0].len();

    labels.into_iter().map(|label| {
        return if labels_len == 1 && network.shape()[0] != 1 {
            let mut output = vec![0.0; network.shape().last().unwrap().clone()];
            output[usize::from(label[0])] = 1.0;
            output
        } else {
            label.into_iter().map(|x| f64::from(x) / 255.0).collect::<Vec<_>>()
        }
    }).collect()
}