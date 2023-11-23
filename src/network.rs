use crate::layer::Layer;

#[derive(Debug)]
pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
}

impl LossFunction {
    fn apply(&self, output: &[f64], target: &[f64]) -> f64 {
        match self {
            Self::MeanSquaredError => mse_loss(output, target),
            Self::CrossEntropy => ce_loss(output, target),
        }
    }
}

fn mse_loss(output: &[f64], target: &[f64]) -> f64 {
    output.iter().zip(target.iter())
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f64>() / output.len() as f64
}

fn ce_loss(output: &[f64], target: &[f64]) -> f64 {
    output.iter().zip(target.iter())
        .map(|(o, t)| -(t * o.ln() + (1.0 - t) * (1.0 - o).ln()))
        .sum::<f64>() / output.len() as f64
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Result<Self, String> {
        for i in 0..layers.len() - 1 {
            if layers[i].output_size() != layers[i + 1].input_size() {
                return Err("Layer sizes are not compatible".to_string());
            }
        }
        Ok(Network { layers })
    }

    pub fn forward(&self, inputs: &[f64]) -> Result<Vec<f64>, String> {
        let result = self.layers
            .iter()
            .fold(inputs.to_vec(), |acc, layer| {
                layer.forward(&acc)
            });
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ActivationFunction::{ReLU, Sigmoid};

    #[test]
    fn test_network_new() {
        let layers = vec![
            Layer::new_mock(1, 2, ReLU),
            Layer::new_mock(2, 1, ReLU),
        ];
        let network = Network::new(layers).unwrap();
        assert_eq!(network.layers.len(), 2, "Network should be initialized with two layers.");
    }

    #[test]
    fn test_network_new_incompatible_layers() {
        let too_many_outputs = vec![
            Layer::new_mock(1, 2, ReLU),
            Layer::new_mock(1, 1, ReLU),
        ];
        assert!(Network::new(too_many_outputs).is_err());

        let too_many_inputs = vec![
            Layer::new_mock(1, 1, ReLU),
            Layer::new_mock(2, 1, ReLU),
        ];
        assert!(Network::new(too_many_inputs).is_err());
    }

    #[test]
    fn test_network_forward_relu() {
        let layers = vec![
            Layer::new_mock(1, 2, ReLU),
            Layer::new_mock(2, 1, ReLU),
        ];
        let network = Network::new(layers).unwrap();
        let inputs = vec![1.0];
        let expected_output = 2.0;
        let actual_output = network.forward(&inputs).unwrap()[0];
        assert!((actual_output - expected_output).abs() < 1e-5, "Actual output {} does not match expected output {}", actual_output, expected_output);
    }

    #[test]
    fn test_network_forward_sigmoid() {
        let layers = vec![
            Layer::new_mock(1, 2, Sigmoid),
            Layer::new_mock(2, 1, Sigmoid),
        ];
        let network = Network::new(layers).unwrap();
        let inputs = vec![1.0];
        let expected_output = 1.0 / (1.0 + (-2.0 / (1.0 + (-1.0f64).exp())).exp());
        let actual_output = network.forward(&inputs).unwrap()[0];
        assert!((actual_output - expected_output).abs() < 1e-5, "Actual output {} does not match expected output {}", actual_output, expected_output);
    }
}
