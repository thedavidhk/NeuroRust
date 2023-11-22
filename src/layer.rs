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
}

fn sigmoid(input: f64) -> f64 {
    1.0 / (1.0 + (-input).exp())
}

fn relu(input: f64) -> f64 {
    input.max(0.0)
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
    activation_function: ActivationFunction,
}

impl Layer {
    pub fn new(num_neurons: usize, inputs_per_neuron: usize, activation_function: ActivationFunction) -> Layer {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::new(inputs_per_neuron))
            .collect();
        Layer {neurons, activation_function}
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter()
            .map(|neuron| neuron.forward(inputs))
            .map(|output| self.activation_function.apply(output))
            .collect()
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
        let layer = Layer::new(num_neurons, inputs_per_neuron, activation_function);

        assert_eq!(layer.neurons.len(), num_neurons, "Number of neurons is incorrect.");

        for neuron in layer.neurons {
            assert_eq!(neuron.test_get_weights().len(), inputs_per_neuron, "Number of neurons is incorrect.");
        }

        match layer.activation_function {
            ActivationFunction::ReLU => {},
            _ => panic!("Layer should have ReLU as its activation function."),
        }
    }

    #[test]
    fn test_layer_forward() {
        let neurons = vec![Neuron::new_mock(), Neuron::new_mock()];
        let activation_function = ActivationFunction::Sigmoid;
        let mut layer = Layer {neurons, activation_function};

        let inputs = vec![1.0];
        let expected_result = 1.0 / (1.0 + (-1.0f64).exp());
        let actual_result = layer.forward(&inputs)[0];
        assert!((actual_result - expected_result).abs() < 1e-5, "Expected and actual results differ with sigmoid.");

        layer.activation_function = ActivationFunction::ReLU;
        let inputs = vec![1.0];
        let expected_result = 1.0;
        let actual_result = layer.forward(&inputs)[0];
        assert!((actual_result - expected_result).abs() < 1e-5, "Expected and actual results differ with ReLU (positive number).");

        let inputs = vec![-1.0];
        let expected_result = 0.0;
        let actual_result = layer.forward(&inputs)[0];
        assert!((actual_result - expected_result).abs() < 1e-5, "Expected and actual results differ with ReLu (negative number).");
    }
}
