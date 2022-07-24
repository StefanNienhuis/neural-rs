# neural-rs

A neural network project written in Rust, written for a school project. 

## Components

 - `neural` - a general purpose neural network library
 - `neural-cli` - a CLI tool for the neural library that handles IDX file formats
 - `neural-emnsit` - a web interface for the neural library handling networks trained with EMNIST data

*Notice:* This is my first large project written in Rust, so there's probably a lot of optimization to be done.

### neural-emnist

`neural-emnist/src` provides a Rust crate with WebAssembly bindings for the neural library.

It can be compiled to WebAssembly with `wasm-pack build --target web`.

`neural-emnist/www` provides a Svelte based web interface that can detect digits and letters. The default network is a network trained with EMNIST digits with a 99.06% accuracy on the test dataset. It uses the `pkg/` directory from the WebAssembly build as a dependency.

After pressing detect, the bounding box of the drawing is calculated and a square is extracted, then down sampled to a 28x28 pixel image using bilinear interpolation. Drawing on a 28x28 canvas directly resulted in lower accuracy, as the digit would not be centered at all times.

## Example usage

### EMNIST digits

Create a new network with 784 inputs and 10 outputs, the layers in between and cost function can be customized.
```shell
neural-cli create -l input:784 relu:300 relu:50 sigmoid:10 -c mean-squared-error ./network.nnet
```

Train the network with the EMNIST digits dataset (epochs: 30, learning rate: 0.3). You can obtain the dataset from [the nist.gov website](https://www.nist.gov/itl/products-and-services/emnist-dataset).
```shell
neural-cli train -e 30 -r 0.3 --test-inputs ./test-images --test-labels ./test-labels ./network.nnet ./train-images ./train-labels
```

Afterwards, open the neural-emnist web interface and upload your newly trained network. Draw a few digits to see how well it's working.

If everything is working, play around with the hyperparameters (layers, epochs, learning rate etc...) a bit to see how this influences the accuracy.

The digits dataset can also be replaced by the letters dataset, for a-z detection. The output layer should be of size 27, as the labels are indexed starting at 1. Output 0 can be ignored.