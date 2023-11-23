use rand::Rng;

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    pub fn new(input_size: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..input_size).map(|_| rng.gen::<f64>()).collect();
        let bias = rng.gen::<f64>();

        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Input size must match weights size."
        );
        let weighted_sum: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, &i)| w * i)
            .sum();
        weighted_sum + self.bias
    }

    pub fn input_size(&self) -> usize {
        self.weights.len()
    }

    #[cfg(test)]
    pub fn test_get_weights(&self) -> &Vec<f64> {
        &self.weights
    }

    #[cfg(test)]
    pub fn new_mock(n_inputs: usize) -> Neuron {
        Neuron {
            weights: vec![1.0; n_inputs],
            bias: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_new() {
        let input_size = 5;
        let neuron = Neuron::new(input_size);

        assert_eq!(
            neuron.weights.len(),
            input_size,
            "Number of weights should match input size."
        );
        assert!(
            neuron.weights.iter().any(|&w| w != 0.0),
            "Weights should not be all zero."
        );
        assert_ne!(neuron.bias, 0.0, "Bias should be initialized.");
    }

    #[test]
    fn test_neuron_forward() {
        let neuron = Neuron {
            weights: vec![-0.5, 0.5],
            bias: 1.0,
        };
        let inputs = vec![1.0, 2.0];
        let expected_output = 1.5;
        let actual_output = neuron.forward(&inputs);
        assert!(
            (actual_output - expected_output).abs() < 1e-5,
            "Forward function does not produce expected output."
        )
    }
}
