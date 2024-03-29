use std::collections::HashMap;


struct GraphInput { 
}
 
fn create_pairs(inputs: Vec<GraphInput>) -> Vec<(GraphInput, GraphInput, f64)> {
     let mut pairs = Vec::new();

    //  Pair each input with every other input
    for i in 0..inputs.len() {
        for j in 0..inputs.len() {
            // Assign a label (1.0 for similar, 0.0 for different)
            // For simplicity, consider same indices as similar
            let label = if i == j { 1.0 } else { 0.0 };

            pairs.push((inputs[i].clone(), inputs[j].clone(), label));
        }
    }

    pairs
}


struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    fn new(weights: Vec<f64>, bias: f64) -> Self {
        Perceptron { weights, bias }
    }

    fn forward(&self, inputs: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(inputs)
            .map(|(w, i)| w * i)
            .sum::<f64>()
            + self.bias
    }

    // Placeholder method for updating weights
    fn update_weights(&mut self, learning_rate: f64, input: &[f64], error: f64) {
        for (w, &i) in self.weights.iter_mut().zip(input.iter()) {
            *w += learning_rate * error * i;
        }
        self.bias += learning_rate * error;
    }
}

struct Graph {
    nodes: HashMap<usize, Perceptron>,
    edges: HashMap<usize, Vec<usize>>, // Representing connections from one node to others
}

impl Graph {
    fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    } 
   fn compare(&self, input1: &GraphInput, input2: &GraphInput) -> f64 {
      
    }

    // Training function needs to be adapted for one-shot learning
    fn train_one_shot(&mut self, pairs: Vec<(GraphInput, GraphInput, f64)>) {
       for (input1, input2, label) in pairs {
            let similarity = self.compare(&input1, &input2);

            // Calculate some form of loss based on the label and similarity
            // For instance, mean squared error (MSE)
            let loss = (label - similarity).powi(2);

            // Backpropagate this loss to update the network
            // This would involve adjusting weights based on the loss
            // Specifics depend on your network architecture
           // ??
 
        }
    }

    
    fn adjust_graph(&mut self, adjustment_condition: bool) {
        //   condition for adjustment
        if adjustment_condition {
            // Example logic for adding a new node
            let new_node_id = self.nodes.len();
            let new_weights = vec![0.5; 2]; // Placeholder weights
            let new_bias = 0.1; // Placeholder bias
            self.add_new_perceptron(new_node_id, new_weights, new_bias);

            // Example logic for connecting the new node
            // Here, we're simply connecting the new node to an existing node
            if new_node_id > 0 {
                self.connect(new_node_id - 1, new_node_id);
            }

            // removing nodes, adjusting weights, etc.
        }
    }

    fn add_perceptron(&mut self, id: usize, perceptron: Perceptron) {
        self.nodes.insert(id, perceptron);
    }

    fn add_new_perceptron(&mut self, id: usize, weights: Vec<f64>, bias: f64) {
        let new_perceptron = Perceptron::new(weights, bias);
        self.add_perceptron(id, new_perceptron);
    }

    fn connect(&mut self, from_id: usize, to_id: usize) {
        self.edges
            .entry(from_id)
            .or_insert_with(Vec::new)
            .push(to_id);
    }
    fn evaluate(&self, inputs: &Vec<Vec<f64>>) -> Vec<f64> {
        inputs
            .iter()
            .map(|input| {
                // Assuming a simplistic evaluation where we use a specific perceptron
                let perceptron_id = 0; // For example, always using the first perceptron
                if let Some(perceptron) = self.nodes.get(&perceptron_id) {
                    perceptron.forward(input)
                } else {
                    0.0 // Default output if the perceptron doesn't exist
                }
            })
            .collect()
    }

    // Placeholder for a more complex propagation mechanism
    fn propagate(&self, start_id: usize, input: &[f64]) {
        if let Some(perceptron) = self.nodes.get(&start_id) {
            let output = perceptron.forward(input);
            if let Some(connections) = self.edges.get(&start_id) {
                for &connected_id in connections {
                    self.propagate(connected_id, &[output]);
                }
            }
        }
    }
}

fn vectorize(token: &str) -> Vec<f64> {
    // Implementation of the function
    //  converting each character of the token to its numeric ASCII value
    token.bytes().map(|c| c as f64).collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

// Function to normalize tokens to lowercase
fn normalize(tokens: &[String]) -> Vec<String> {
    tokens.iter()
        .map(|s| s.to_lowercase())
        .collect()
}


fn main() {
    let mut graph = Graph::new();

    //  adding perceptrons and connecting them
    graph.add_perceptron(0, Perceptron::new(vec![1.0, 2.0], 0.5));
    graph.add_perceptron(1, Perceptron::new(vec![3.0, 4.0], 1.0));
    graph.connect(0, 1);

    // Start propagation from node 0 with some input
    graph.propagate(0, &[0.5, 1.5]);

    let text = "Hello, World! This is an example text.";

    // Tokenize and normalize the text
    let tokens = tokenize(text);
    let normalized_tokens = normalize(&tokens);

    // Initialize a vector to collect outputs from the graph
    let mut graph_outputs: Vec<f64> = Vec::new();

    // Vectorize and process each token
    for token in normalized_tokens.iter() {
        let vectorized_token = vectorize(token);
        println!("Token: {:?}, Vectorized: {:?}", token, vectorized_token);

        // Hypothetical logic to decide which perceptron to start with
        // For simplicity, let's always start with perceptron 0
        let start_perceptron_id = 0;

        // Feed the vectorized token into the graph and collect the output
        graph.propagate(start_perceptron_id, &vectorized_token);

        // Assuming the output of the propagation is the last perceptron activated
        // This is a simplification; actual implementation might differ
        if let Some(last_perceptron) = graph.nodes.get(&start_perceptron_id) {
            let output = last_perceptron.forward(&vectorized_token);
            graph_outputs.push(output);
        }
    }

    // Post-processing the graph outputs
    // This could be any kind of logic, depending on your application
    // ?? making a decision based on the outputs, aggregating them, etc.
    println!("Graph Outputs: {:?}", graph_outputs);

    let learning_rate = 0.01;
    let input = vec![0.5, 1.5]; // Example input
    let error = 0.1; // Hypothetical error
    if let Some(perceptron) = graph.nodes.get_mut(&0) {
        perceptron.update_weights(learning_rate, &input, error);
    }
    let transformed_outputs_for_evaluation: Vec<Vec<f64>> = graph_outputs
    .iter()
    .map(|&output| vec![output])
    .collect();

    let re_evaluation_results = graph.evaluate(&transformed_outputs_for_evaluation);
    println!("Re-Evaluation Results: {:?}", re_evaluation_results);

    //  adding a new perceptron to the graph
    let new_weights = vec![0.3, 0.7];
    let new_bias = 0.2;
    graph.add_new_perceptron(2, new_weights, new_bias); 
    graph.adjust_graph(true);
}
