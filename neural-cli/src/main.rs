mod io;
mod idx;

use neural::network::Network;
use bincode;
use std::path::PathBuf;
use std::{process};
use clap::{Parser, Subcommand};
use rand::{seq::SliceRandom};

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

        #[clap(short='r', long)]
        learning_rate: f64,

        /// The batch size
        #[clap(short='s', long, default_value = "10")]
        batch_size: usize,

        /// The number of batches to train, defaulting to all batches
        #[clap(short='c', long)]
        batch_count: Option<usize>,
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
        Commands::Create { network } => create(network),
        Commands::Train { network, images, labels, learning_rate, batch_size, batch_count } => train(network, images, labels, learning_rate, batch_size, batch_count, cli.verbose),
        Commands::Test { network, images, labels, count } => test(network, images, labels, count, cli.verbose),
        Commands::Evaluate { network, image } => evaluate(network, image)
    }
}

fn create(network_path: &PathBuf) {
    let network = Network::random(vec![784, 30, 10]);

    let encoded: Vec<u8> = bincode::encode_to_vec(network, bincode::config::standard()).unwrap();

    match io::write_file(&encoded, network_path, true, Some("nnet")) {
        Err(error) => {
            println!("Couldn't write to {}: {}", network_path.display(), error);
            return;
        },
        Ok(_) => println!("Created a new neural network at {}", network_path.display())
    };
}

fn train(network_path: &PathBuf, images_path: &PathBuf, labels_path: &PathBuf, learning_rate: &f64, batch_size: &usize, batch_count: &Option<usize>, verbose: bool) {
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

    let training_data: Vec<(Vec<f64>, Vec<f64>)> =
        images.images.iter()
            // Create 1-dimensional f64 Vec from 2-dimensional u8 Vec
            .map(|image| image.iter().flatten().map(|x| f64::from(*x)).collect::<Vec<f64>>())
            // Zip with expected output - expected output is a Vec of length 10 with only the expected output set to 1.0
            .zip(labels.labels.iter().map(|x| {
                // one liner?
                let mut output = vec![0.0; 10];
                output[usize::from(*x)] = 1.0;
                return output;
            }))
            .take(
                match batch_count {
                    Some(batch_count) => batch_size * batch_count,
                    None => images.image_count as usize
                }
            )
            .collect();

    println!(
        "Starting training with {} batches of size {} and {} total samples",
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

    let mut test_data: Vec<(&Vec<Vec<u8>>, &u8)> = images.images.iter().zip(labels.labels.iter()).collect();
    let mut rng = rand::thread_rng();

    test_data.shuffle(&mut rng);

    let test_batch: Vec<&(&Vec<Vec<u8>>, &u8)> = test_data.iter().take(count).collect();

    let mut correct_count = 0;

    for (image, label) in test_batch {
        let pixels: Vec<f64> = image.iter().flatten().map(|x| f64::from(*x) / 255.0).collect();
        let label: u8 = **label;

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

    println!("Accuracy: {}/{} - {:.2}%", correct_count, count, f64::from(correct_count) / (count as f64) * 100.0);
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

    if image_data.len() != *network.shape.first().unwrap_or(&0) {
        println!("Incorrect image data length ({}) should be {}", image_data.len(), network.shape.first().unwrap_or(&0));
        return;
    }

    let output = network.feed_forward(image_data);

    for (i, p) in output.iter().enumerate() {
        println!("{:>2}: {:.2}%", i + 1, p * 100.0);
    }
}