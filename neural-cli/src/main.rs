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
    verbose: bool
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new neural network
    Create {
        /// The neural network file
        #[clap(short, long, value_name = "FILE")]
        network: PathBuf,

        /// A layer that should be added to the network
        ///
        /// Specified with [activation_function]:[size].
        /// Must start with input:[size]
        #[clap(short, long="layer", action(ArgAction::Append), min_values(2), required(true))]
        layers: Vec<String>
    },

    /// Train a neural network with the provided data set
    Train {
        /// The neural network file
        #[clap(short, long, value_name = "FILE")]
        network: PathBuf,

        /// The image dataset file
        #[clap(short, long, value_name = "FILE")]
        images: PathBuf,

        /// The label dataset file
        #[clap(short, long, value_name = "FILE")]
        labels: PathBuf,

        /// The learning rate
        #[clap(short='r', long)]
        learning_rate: f64,

        /// The batch size
        #[clap(short='s', long, default_value = "10")]
        batch_size: usize,

        /// The number of batches to train, defaulting to all batches
        #[clap(short='c', long)]
        batch_count: Option<usize>,

        /// The amount of times it should train
        #[clap(short, long, default_value = "1")]
        epochs: usize,

        /// If provided, tests the network with these images at every epoch
        #[clap(long, requires("test-labels"))]
        test_images: Option<PathBuf>,

        /// If provided, tests the network with these labels at every epoch
        #[clap(long, requires("test-images"))]
        test_labels: Option<PathBuf>
    },

    /// Test a neural network with the provided data set
    Test {
        /// The neural network file
        #[clap(short, long, value_name = "FILE")]
        network: PathBuf,

        /// The image dataset file
        #[clap(short, long, value_name = "FILE")]
        images: PathBuf,

        /// The label dataset file
        #[clap(short, long, value_name = "FILE")]
        labels: PathBuf,

        /// The amount of samples to test
        #[clap(short, long)]
        count: Option<usize>
    },

    /// Evaluate a neural network with the provided image
    Evaluate {
        /// The neural network file
        #[clap(short, long, value_name = "FILE")]
        network: PathBuf,

        /// The image file
        #[clap(short, long, value_name = "FILE")]
        image: PathBuf,
    }

}

fn main() {
    ctrlc::set_handler(|| {
        process::exit(0);
    }).expect("Error while settings Ctrl-C handler");

    let cli = Cli::parse();

    match &cli.command {
        Commands::Create { network, layers } => create(network, layers),
        Commands::Train { network, images, labels, learning_rate, batch_size, batch_count, epochs, test_images, test_labels } => train(network, images, labels, learning_rate, batch_size, batch_count, epochs, test_images, test_labels, cli.verbose),
        Commands::Test { network, images, labels, count } => test(network, images, labels, count, cli.verbose),
        Commands::Evaluate { network, image } => evaluate(network, image)
    }
}

fn create(network_path: &PathBuf, layers: &[String]) {
    let mut network = Network::new();

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

fn train(network_path: &PathBuf, images_path: &PathBuf, labels_path: &PathBuf,
         learning_rate: &f64, batch_size: &usize, batch_count: &Option<usize>, epochs: &usize,
         test_images: &Option<PathBuf>, test_labels: &Option<PathBuf>, verbose: bool) {

    let mut network = match io::parse_network_file(network_path) {
        Err(error) => {
            println!("Error while reading network: {}", error);
            return;
        }
        Ok(network) => network
    };

    let images = match io::parse_idx_image_file(images_path) {
        Err(error) => {
            println!("Error while reading images: {}", error);
            return;
        }
        Ok(images) => images
    };

    let labels = match io::parse_idx_label_file(labels_path) {
        Err(error) => {
            println!("Error while reading labels: {}", error);
            return;
        }
        Ok(labels) => labels
    };

    let test_images = match test_images {
        Some(test_images) => match io::parse_idx_image_file(test_images) {
            Err(error) => {
                println!("Error while reading test images: {}", error);
                return;
            }
            Ok(images) => Some(images)
        },
        None => None
    };

    let test_labels = match test_labels {
        Some(test_labels) => match io::parse_idx_label_file(test_labels) {
            Err(error) => {
                println!("Error while reading test labels: {}", error);
                return;
            }
            Ok(labels) => Some(labels)
        }
        None => None
    };

    let mut test_data = match (test_images, test_labels) {
        (Some(test_images), Some(test_labels)) => Some(test_images.images.iter().cloned().zip(test_labels.labels.iter().cloned()).collect::<Vec<(Vec<Vec<u8>>, u8)>>()),
        _ => None
    };

    let mut rng = rand::thread_rng();

    println!();

    for i in 0..*epochs {
        let mut training_data: Vec<(Vec<f64>, Vec<f64>)> =
            images.images.iter()
                // Create 1-dimensional f64 Vec from 2-dimensional u8 Vec
                .map(|image| image.iter().flatten().map(|x| f64::from(*x) / 255.0).collect::<Vec<f64>>())
                // Zip with expected output - expected output is a Vec of length 10 with only the expected output set to 1.0
                .zip(labels.labels.iter().map(|x| {
                    // one liner?
                    let mut output = vec![0.0; 10];
                    output[usize::from(*x)] = 1.0;
                    return output;
                }))
                .collect();

        training_data.shuffle(&mut rng);

        training_data = training_data.iter().take(
            match batch_count {
                Some(batch_count) => batch_size * batch_count,
                None => images.image_count as usize
            }
        ).map(|x| x.clone()).collect();

        println!(
            "Starting epoch {} training with {} batches of size {} and {} total samples",
            i,
            match batch_count {
                Some(batch_count) => batch_count.to_string(),
                None => "all".to_string()
            },
            batch_size,
            training_data.len()
        );

        let training_data_len = training_data.len();

        network.stochastic_gradient_descent(training_data, *batch_size, *learning_rate, |progress| {
            if verbose {
                println!("{}/{} {:.2}%", progress, training_data_len / batch_size, (progress as f64 * *batch_size as f64) / (training_data_len as f64) * 100.0);
            }
        });

        println!("Finished training for epoch {}.", i);

        if let Some(test_data) = &mut test_data {
            test_data.shuffle(&mut rng);
            let correct_count = test_only(&network, test_data, verbose);

            println!("Accuracy: {}/{} - {:.2}%", correct_count, test_data.len(), f64::from(correct_count) / (test_data.len() as f64) * 100.0);
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

fn test(network_path: &PathBuf, images_path: &PathBuf, labels_path: &PathBuf, count: &Option<usize>, verbose: bool) {
    let network = match io::parse_network_file(network_path) {
        Err(error) => {
            println!("Error while reading network: {}", error);
            return;
        }
        Ok(network) => network
    };

    let images = match io::parse_idx_image_file(images_path) {
        Err(error) => {
            println!("Error while reading images: {}", error);
            return;
        }
        Ok(images) => images
    };

    let labels = match io::parse_idx_label_file(labels_path) {
        Err(error) => {
            println!("Error while reading labels: {}", error);
            return;
        }
        Ok(labels) => labels
    };

    let count = count.unwrap_or(images.image_count as usize);

    let mut test_data: Vec<(Vec<Vec<u8>>, u8)> = images.images.iter().cloned().zip(labels.labels.iter().cloned()).collect();
    let mut rng = rand::thread_rng();

    test_data.shuffle(&mut rng);

    let test_batch: Vec<(Vec<Vec<u8>>, u8)> = test_data.iter().take(count).cloned().collect();

    let correct_count = test_only(&network, &test_batch, verbose);

    println!("Accuracy: {}/{} - {:.2}%", correct_count, count, f64::from(correct_count) / (count as f64) * 100.0);
}

fn test_only(network: &Network, test_batch: &[(Vec<Vec<u8>>, u8)], verbose: bool) -> i32 {
    let mut correct_count = 0;

    for (image, label) in test_batch {
        let pixels: Vec<f64> = image.iter().flatten().map(|x| f64::from(*x) / 255.0).collect();
        let label: u8 = *label;

        let result = network.feed_forward(pixels);
        let (i, p) = result.iter().enumerate().max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap()).unwrap();

        if i != usize::from(label) {
            if verbose {
                println!("Wrong: {} = {} @ {:.2}%", label, i, p * 100.0);
            }
        } else {
            if verbose {
                println!("Correct: {} = {} @ {:.2}%", label, i, p * 100.0);
            }

            correct_count += 1;
        }
    }

    return correct_count;
}

fn evaluate(network_path: &PathBuf, image_path: &PathBuf) {
    let network = match io::parse_network_file(network_path) {
        Err(error) => {
            println!("{}", error);
            return;
        }
        Ok(network) => network
    };

    let image_data: Vec<f64> = match io::read_file(image_path) {
        Err(error) => {
            println!("Error while reading image: {}", error);
            return;
        }
        Ok(network) => network.iter().map(|x| f64::from(*x) / 255.0).collect()
    };

    if image_data.len() != *network.shape().first().unwrap_or(&0) {
        println!("Incorrect image data length ({}) should be {}", image_data.len(), network.shape().first().unwrap_or(&0));
        return;
    }

    let output = network.feed_forward(image_data);

    for (i, p) in output.iter().enumerate() {
        println!("{:>2}: {:.2}%", i + 1, p * 100.0);
    }
}