use crate::neuron::Neuron;

#[derive(Debug)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
}

impl ActivationFunction {
    fn apply(&self, input: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => sigmoid(input),
            ActivationFunction::ReLU => relu(input),
        }
    }

    fn derivative(&self, input: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => d_sigmoid(input),
            ActivationFunction::ReLU => d_relu(input),
        }
    }
}

fn sigmoid(input: f64) -> f64 {
    1.0 / (1.0 + (-input).exp())
}

fn d_sigmoid(input: f64) -> f64 {
    let intermediate = sigmoid(input);
    intermediate * (1.0 - intermediate)
}

fn relu(input: f64) -> f64 {
    input.max(0.0)
}

fn d_relu(input: f64) -> f64 {
    (input >= 0.0) as i64 as f64
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
    activation_function: ActivationFunction,
}

impl Layer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation_function: ActivationFunction,
    ) -> Layer {
        let neurons = (0..output_size).map(|_| Neuron::new(input_size)).collect();
        Layer {
            neurons,
            activation_function,
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        // TODO: Error handling with results instead of asserts
        assert_eq!(
            inputs.len(),
            self.input_size(),
            "Input size must match layer inputs."
        );
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .map(|output| self.activation_function.apply(output))
            .collect()
    }

    pub fn input_size(&self) -> usize {
        self.neurons[0].input_size()
    }

    pub fn output_size(&self) -> usize {
        self.neurons.len()
    }

    pub fn update_weights(&self, inputs: &[f64], deltas: &[f64], learning_rate: f64) {
        self.neurons.iter().enumerate().map(|(i, neuron)| {
            let gradient = inputs[i] * deltas[i];
            neuron.update_weights(gradient, learning_rate);
        }).collect()
    }

    pub fn update_bias(&self, deltas: &[f64], learning_rate: f64) {
        for (neuron, &delta) in self.neurons.iter_mut().zip(deltas.iter()) {
            neuron.update_bias(delta, learning_rate);
        }
    }

    pub fn get_deltas_for_previous_layer(&self, activations: &[f64], deltas: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|neuron| {
            neuron.weights.iter().zip(deltas.iter())
                .map(|(&weight, &delta)| weight * delta)
                .sum::<f64>() * derivative_of_activation_function(neuron.output)
        }).collect()
    }

    #[cfg(test)]
    pub fn new_mock(
        input_size: usize,
        output_size: usize,
        activation_function: ActivationFunction,
    ) -> Layer {
        let neurons = (0..output_size)
            .map(|_| Neuron::new_mock(input_size))
            .collect();
        Layer {
            neurons,
            activation_function,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_new() {
        let num_neurons = 5;
        let inputs_per_neuron = 1;
        let activation_function = ActivationFunction::ReLU;
        let layer = Layer::new(inputs_per_neuron, num_neurons, activation_function);

        assert_eq!(
            layer.neurons.len(),
            num_neurons,
            "Number of neurons is incorrect."
        );

        for neuron in layer.neurons {
            assert_eq!(
                neuron.test_get_weights().len(),
                inputs_per_neuron,
                "Number of neurons is incorrect."
            );
        }

        match layer.activation_function {
            ActivationFunction::ReLU => {}
            _ => panic!("Layer should have ReLU as its activation function."),
        }
    }

    #[test]
    fn test_layer_forward() {
        let neurons = vec![Neuron::new_mock(1), Neuron::new_mock(1)];
        let activation_function = ActivationFunction::Sigmoid;
        let mut layer = Layer {
            neurons,
            activation_function,
        };

        let inputs = vec![1.0];
        let expected_result = 1.0 / (1.0 + (-1.0f64).exp());
        let actual_result = layer.forward(&inputs)[0];
        assert!(
            (actual_result - expected_result).abs() < 1e-5,
            "Expected and actual results differ with sigmoid."
        );

        layer.activation_function = ActivationFunction::ReLU;
        let inputs = vec![1.0];
        let expected_result = 1.0;
        let actual_result = layer.forward(&inputs)[0];
        assert!(
            (actual_result - expected_result).abs() < 1e-5,
            "Expected and actual results differ with ReLU (positive number)."
        );

        let inputs = vec![-1.0];
        let expected_result = 0.0;
        let actual_result = layer.forward(&inputs)[0];
        assert!(
            (actual_result - expected_result).abs() < 1e-5,
            "Expected and actual results differ with ReLu (negative number)."
        );
    }
}
