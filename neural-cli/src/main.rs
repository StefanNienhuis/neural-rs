mod io;
mod idx;

use neural::network::Network;
use bincode;
use std::path::PathBuf;
use std::fs::File;
use std::io::Write;
use clap::{Parser, Subcommand};

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
        #[clap(short, long, value_parser, value_name = "FILE")]
        network: PathBuf,
    },

    // Train a neural network with the provided data set
    Train {
        // The neural network file
        #[clap(short, long, value_parser, value_name = "FILE")]
        network: PathBuf,

        // The image dataset file
        #[clap(short, long, value_parser, value_name = "FILE")]
        images: PathBuf,

        // The label dataset file
        #[clap(short, long, value_parser, value_name = "FILE")]
        labels: PathBuf,
    },

    // Evaluate a neural network with the provided image
    Evaluate {
        // The neural network file
        #[clap(short, long, value_parser, value_name = "FILE")]
        network: PathBuf,

        // The image file
        #[clap(short, long, value_parser, value_name = "FILE")]
        image: PathBuf,
    }

}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Create { network } => create(network),
        Commands::Train { network, images, labels } => train(network, images, labels),
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