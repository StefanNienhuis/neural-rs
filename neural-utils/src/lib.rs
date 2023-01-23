pub mod io;
pub mod idx;
use neural::{Network, Float};

pub fn outputs_from_labels(network: &Network, labels: Vec<Vec<u8>>) -> Vec<Vec<Float>> {
    let labels_len = labels[0].len();

    labels
        .into_iter()
        .map(|label| {
            return if labels_len == 1 && network.shape()[0] != 1 {
                let mut output = vec![0.0; network.shape().last().unwrap().clone()];
                output[usize::from(label[0])] = 1.0;
                output
            } else {
                label
                    .into_iter()
                    .map(|x| Float::from(x) / 255.0)
                    .collect::<Vec<_>>()
            };
        })
        .collect()
}