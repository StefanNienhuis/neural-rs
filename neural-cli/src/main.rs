mod io;
mod idx;

use std::cmp::Ordering;
use neural::network::Network;
use bincode;
use std::path::PathBuf;
use std::fs::File;
use std::io::Write;
use clap::{Parser, Subcommand};
use rand::{seq::SliceRandom};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands
}

#[derive(Subcommand)]
enum Commands {
    // Create a new neural network
    Create {
        // The neural network file
        #[clap(short, long, value_name = "FILE")]
        network: PathBuf,
    },

    // Train a neural network with the provided data set
    Train {
        // The neural network file
        #[clap(short, long, value_name = "FILE")]
        network: PathBuf,

        // The image dataset file
        #[clap(short, long, value_name = "FILE")]
        images: PathBuf,

        // The label dataset file
        #[clap(short, long, value_name = "FILE")]
        labels: PathBuf,
    },

    // Test a neural network with the provided data set
    Test {
        // The neural network file
        #[clap(short, long, value_name = "FILE")]
        network: PathBuf,

        // The image dataset file
        #[clap(short, long, value_name = "FILE")]
        images: PathBuf,

        // The label dataset file
        #[clap(short, long, value_name = "FILE")]
        labels: PathBuf,

        // The amount of samples to test
        #[clap(short, long, default_value = "10")]
        count: usize
    },

    // Evaluate a neural network with the provided image
    Evaluate {
        // The neural network file
        #[clap(short, long, value_name = "FILE")]
        network: PathBuf,

        // The image file
        #[clap(short, long, value_name = "FILE")]
        image: PathBuf,
    }

}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Create { network } => create(network),
        Commands::Train { network, images, labels } => train(network, images, labels),
        Commands::Test { network, images, labels, count } => test(network, images, labels, count),
        Commands::Evaluate { network, image } => evaluate(network, image)
    }
}

fn create(network_path: &PathBuf) {
    if network_path.exists() {
        println!("{} already exists", network_path.display());
        return;
    } else if network_path.extension().is_none() {
        println!("{} should end with 'nnet' extension", network_path.display());
        return;
    } else if network_path.extension().unwrap() != "nnet" {
        println!("{} should end with 'nnet' extension", network_path.display());
        return;
    }

    let network = Network::random(vec![784, 30, 10]);

    let encoded: Vec<u8> = bincode::encode_to_vec(network, bincode::config::standard()).unwrap();

    let mut file = match File::create(network_path) {
        Err(error) => {
            println!("Couldn't create {}: {}", network_path.display(), error);
            return;
        },
        Ok(file) => file
    };

    match file.write_all(encoded.as_slice()) {
        Err(error) => {
            println!("Couldn't write to {}: {}", network_path.display(), error);
            return;
        },
        Ok(_) => println!("Created a new neural network at {}", network_path.display())
    };
}

fn train(network_path: &PathBuf, images_path: &PathBuf, labels_path: &PathBuf) {
    let network = match io::parse_network_file(network_path) {
        Err(error) => {
            println!("{}", error);
            return;
        }
        Ok(network) => network
    };

    let images = match io::parse_idx_image_file(images_path) {
        Err(error) => {
            println!("{}", error);
            return;
        }
        Ok(images) => images
    };

    let labels = match io::parse_idx_label_file(labels_path) {
        Err(error) => {
            println!("{}", error);
            return;
        }
        Ok(labels) => labels
    };

    let result = network.feed_forward(vec![0.5; 784]);

    println!("{:?} {} {}", result, images.images.len(), labels.labels.len())
}

fn test(network_path: &PathBuf, images_path: &PathBuf, labels_path: &PathBuf, count: &usize) {
    let network = match io::parse_network_file(network_path) {
        Err(error) => {
            println!("{}", error);
            return;
        }
        Ok(network) => network
    };

    let images = match io::parse_idx_image_file(images_path) {
        Err(error) => {
            println!("{}", error);
            return;
        }
        Ok(images) => images
    };

    let labels = match io::parse_idx_label_file(labels_path) {
        Err(error) => {
            println!("{}", error);
            return;
        }
        Ok(labels) => labels
    };

    let mut test_data: Vec<(&Vec<Vec<u8>>, &u8)> = images.images.iter().zip(labels.labels.iter()).collect();
    let mut rng = rand::thread_rng();

    test_data.shuffle(&mut rng);

    let test_batch: Vec<&(&Vec<Vec<u8>>, &u8)> = test_data.iter().take(*count).collect();

    let mut correct_count = 0;

    for (image, label) in test_batch {
        let pixels: Vec<f64> = image.iter().flatten().map(|x| f64::from(*x) / 255.0).collect();
        let label: u8 = **label;

        let result = network.feed_forward(pixels);
        let (i, p) = result.iter().enumerate().max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap()).unwrap();

        if i != usize::from(label) {
            println!("Wrong: {} = {} @ {:.2}%", label, i, p * 100.0);
        } else {
            println!("Correct: {} = {} @ {:.2}%", label, i, p * 100.0);
            correct_count += 1;
        }
    }

    println!("Accuracy: {}/{} - {:.2}%", correct_count, count, f64::from(correct_count) / (*count as f64) * 100.0);
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