#include <includes/common.h>


/* Implementation of the sigmoid function */
static double sigmoid_function(double input) {
	return 1 / (1 + exp(-input));
}

/* The derivative of sigmoid */
static double sigmoid_derivative(double input) {
	double sig = sigmoid_function(input);
	return sig * (1 - sig);
}

activation_function activation_functions[] = {(activation_function){.activation=sigmoid_function, .activation_derivative=sigmoid_derivative}};

neural_network* init_neural_network(bool* recurrent_layer, size_t* layer_sizes, size_t num_layers) {


	/* Allocate our neural network */
	neural_network* network = malloc(sizeof(neural_network));
	if(!network) {
		error("Failed to allocate neural network.");
		return NULL;
	}

	/* Set the number of layers */
	network->num_layers = num_layers;
	
	/* Allocate space for them */
	network->layers = malloc(sizeof(layer) * num_layers);
	if(!network->layers) {
		error("Failed to allocate neural network layers.");
		return NULL;
	}

	/* Loop through each layer */
	for(int i = 0; i < num_layers; i += 1) {

		layer* network_layer = &network->layers[i];

		/* Allocate room for the neurons in each layer */
		network_layer->layer_neurons = malloc(sizeof(neuron) * layer_sizes[i]);

		if(!network_layer->layer_neurons) {
			error("Failed to allocate neural network neurons.");
			return NULL;
		}

		/* Clear the neurons */
		bzero(network_layer->layer_neurons, sizeof(neuron) * layer_sizes[i]);

		/* Setup the neurons */
		for(int j = 0; j < layer_sizes[i]; j += 1) {
			neuron* curr_neuron = &network_layer->layer_neurons[j];

			/* Set the activation function index */
			curr_neuron->activation_index = 0;

			/* If this layer has a previous layer */
			if(i > 0) {
				/* Set our bias */
				curr_neuron->bias = uniform_decimal();

				/* Allocate an array for the neurons weights */
				curr_neuron->weights = malloc(sizeof(double) * layer_sizes[i-1]);
				if(!curr_neuron->weights) {
					error("Failed to allocate neuron weights.");
					return NULL;
				}

				/* Allocate an array for the neurons weight derivatives */
				curr_neuron->weight_derivatives = malloc(sizeof(double) * layer_sizes[i-1]);
				if(!curr_neuron->weight_derivatives) {
					error("Failed to allocate neuron weight derivatives.");
					return NULL;
				}

				for(int k = 0; k < layer_sizes[i-1]; k += 1) {

					/* Randomise all the weights */
					curr_neuron->weights[k] = uniform_decimal();
				}
			}
		}

		/* Set the number of neurons in each layer and whether it's recurrent */
		network_layer->num_neurons = layer_sizes[i];
		network_layer->recurrent = recurrent_layer[i];

	}

	return network;

}

void free_neural_network(neural_network* network) {

	/* Loop through all the layers */
	for(int i = 0; i < network->num_layers; i += 1) {
		layer* curr_layer = &network->layers[i];

		/* Loop through all the neurons */
		for(int j = 0; j < curr_layer->num_neurons; j += 1) {
			neuron* curr_neuron = &curr_layer->layer_neurons[j];

			/* Deallocate the neurons weights */
			if(curr_neuron->weights != NULL) {
				free(curr_neuron->weights);
				free(curr_neuron->weight_derivatives);
			}
		}

		/* Free the neurons for this layer */
		free(curr_layer->layer_neurons);
	}

	/* Free the network layers and the network */
	free(network->layers);
	free(network);
}

neural_network* import_neural_network(char* filename) {
	size_t file_length;
	char* file_buf;

	int ret = read_file(filename, &file_buf, &file_length);

	if(ret != 0) {
		return NULL;
	}

	/* Verify we can fit a file header at least */
	if(file_length < sizeof(neural_network_file_header)) {
		error("File too small\n");
		free(file_buf);
		return NULL;
	}

	/* Get our file header */
	neural_network_file_header* header = (neural_network_file_header*)file_buf;

	/* Verify the magic */
	if(header->magic != NEURAL_NETWORK_MAGIC) {
		error("Wrong file magic\n");
		free(file_buf);
		return NULL;
	}

	/* Allocate space for our recurrent layer booleans, the layer sizes and the offsets of each layer in the file */
	bool* recurrent_layer = malloc(sizeof(bool) * header->num_layers);
	if(!recurrent_layer) {
		error("Failed to allocate recurrent layer buffer\n");
		free(file_buf);
		return NULL;
	}

	size_t* layer_sizes = malloc(sizeof(size_t) * header->num_layers);
	if(!layer_sizes) {
		error("Failed to allocate recurrent layer buffer\n");
		free(file_buf);
		free(recurrent_layer);
		return NULL;
	}

	size_t* file_layer_offsets = malloc(sizeof(size_t) * header->num_layers);
	if(!file_layer_offsets) {
		error("Failed to allocate recurrent layer buffer\n");
		free(file_buf);
		free(recurrent_layer);
		free(layer_sizes);
		return NULL;
	}


	size_t file_offset = sizeof(neural_network_file_header);

	for(int i = 0; i < header->num_layers; i += 1) {

		/* Verify that our layer isn't larger than the file */
		if((file_offset + sizeof(file_layer)) > file_length) {
			error("File malformed: Not enough space for all layers\n");
			free(recurrent_layer);
			free(layer_sizes);
			free(file_layer_offsets);
			free(file_buf);
			return NULL;
		}

		/* Save the layers offset */
		file_layer_offsets[i] = file_offset;

		file_layer* curr_file_layer = (file_layer*)&file_buf[file_offset];

		/* Get our new layer parameters */
		recurrent_layer[i] = curr_file_layer->recurrent;
		layer_sizes[i] = curr_file_layer->num_neurons;

		/* Move onto the next layer */
		file_offset += curr_file_layer->layer_len;
	}

	/* Initialise a neural network with our settings */
	neural_network* network = init_neural_network(recurrent_layer, layer_sizes, header->num_layers);

	/* Free our settings arrays */
	free(recurrent_layer);
	free(layer_sizes);

	if(!network) {
			error("Failed to allocate neural network\n");
			free(file_layer_offsets);
			free(file_buf);
			return NULL;
	}

	/* Loop through each layer except the first (as weights and biases in this layer are irrelivant) */
	for(int i = 1; i < header->num_layers; i += 1) {
		layer* curr_layer = &network->layers[i];

		size_t file_offset = file_layer_offsets[i] + sizeof(file_layer);

		for(int j = 0; j < curr_layer->num_neurons; j += 1) {

			size_t end_offset = file_offset + sizeof(file_neuron);

			/* Ensure the end of our neuron isn't past the end of the file */
			if(end_offset > file_length) {
				error("File malformed: not enough space for all neurons\n");
				free(file_layer_offsets);
				free(file_buf);
				free_neural_network(network);
				return NULL;
			}

			/* If we're not the last layer, make sure our neuron isn't overwriting the next layer */
			if(i < (header->num_layers - 1) && end_offset > file_layer_offsets[i + 1]) {
				error("File malformed: neurons go past end of layer\n");
				free(file_layer_offsets);
				free(file_buf);
				free_neural_network(network);
				return NULL;
			}

			neuron* curr_neuron = &curr_layer->layer_neurons[j];

			file_neuron* curr_file_neuron = (file_neuron*)&file_buf[file_offset];

			/* Set our bias, recurrent weight and activation inde */
			curr_neuron->bias = curr_file_neuron->bias;
			curr_neuron->recurrent_weight = curr_file_neuron->recurrent_weight;
			curr_neuron->activation_index = curr_file_neuron->activation_index;

			/* Make sure the number of weights is equal to the number of neurons */
			if(curr_file_neuron->num_weights != network->layers[i-1].num_neurons) {
				error("File malformed: number of weights not equal to the number of neurons\n");
				free(file_layer_offsets);
				free(file_buf);
				free_neural_network(network);
				return NULL;
			}

			/* Update the end offset with the number of weights we now know of */
			end_offset += curr_file_neuron->num_weights * sizeof(double);

			/* Make sure the end of our neuron isn't overwriting the next neuron */
			if(end_offset > (file_offset + curr_file_neuron->neuron_len)) {
				error("File malformed: neuron overwriting next neuron\n");
				free(file_layer_offsets);
				free(file_buf);
				free_neural_network(network);
				return NULL;
			}

			/* Make sure the end of our neuron isn't over the file length */
			if(end_offset > file_length) {
				error("File malformed: not enough space for neuron weights\n");
				free(file_layer_offsets);
				free(file_buf);
				free_neural_network(network);
				return NULL;
			}

			/* If we're not the last layer, make sure our neuron isn't overwriting the next layer */
			if(i < (header->num_layers - 1) && end_offset > file_layer_offsets[i + 1]) {
				error("File malformed: neuron weights go past end of layer\n");
				free(file_layer_offsets);
				free(file_buf);
				free_neural_network(network);
				return NULL;
			}

			/* copy over our weights */
			double* weights = (double*)&file_buf[file_offset + sizeof(file_neuron)];
			memcpy(curr_neuron->weights, weights, curr_file_neuron->num_weights * sizeof(double));

			file_offset += curr_file_neuron->neuron_len;
		}
	}

	free(file_buf);
	free(file_layer_offsets);
	return network;
}

void export_neural_network(neural_network* network, char* filename) {

	/* Open our output file */
	FILE* f = fopen(filename, "wb");

	if(!f) {
		error("Failed to open output file\n");
		return;
	}

	/* Start by setting up our file header */
	neural_network_file_header header;
	header.magic = NEURAL_NETWORK_MAGIC;
	header.num_layers = network->num_layers;

	/* Write the header to the file */
	fwrite(&header, sizeof(neural_network_file_header), 1, f);

	for(int i = 0; i < network->num_layers; i += 1) {

		/* Setup the current layer header */
		layer* curr_layer = &network->layers[i];

		file_layer layer;
		layer.num_neurons = curr_layer->num_neurons;
		layer.recurrent = curr_layer->recurrent;

		/* Calculate the size of all our neurons */
		size_t neuron_len = sizeof(file_neuron);

		if(i > 0) {
			neuron_len += sizeof(double) * network->layers[i-1].num_neurons;
		}

		layer.layer_len = neuron_len * curr_layer->num_neurons + sizeof(file_layer);

		/* Write the layer to the file */
		fwrite(&layer, sizeof(file_layer), 1, f);

		/* Loop through all the neurons in our layer */
		for(int j = 0; j < curr_layer->num_neurons; j += 1) {

			/* Get the neuron we're working with, and the position of our new neuron in the array */
			neuron* curr_neuron = &curr_layer->layer_neurons[j];
			file_neuron neuron_header;

			/* Define bias and recurrent weight */
			neuron_header.bias = curr_neuron->bias;
			neuron_header.recurrent_weight = curr_neuron->recurrent_weight;

			/* Set activation index */
			neuron_header.activation_index = curr_neuron->activation_index;
			
			/* Define neuron length */
			neuron_header.neuron_len = neuron_len;


			/* Define the number of weights */
			neuron_header.num_weights = 0;

			if(i > 0) {
				/* The number of weights is the number of neurons in the previous layer */
				neuron_header.num_weights = network->layers[i-1].num_neurons;
			}

			fwrite(&neuron_header, sizeof(file_neuron), 1, f);


	
			if(i > 0) {
				/* Write our weights into the file */
				fwrite(curr_neuron->weights, neuron_header.num_weights, sizeof(double), f);
			}
		}
	}

	fclose(f);
}

static void reset_history(neural_network* network) {

	/* Loop through the network layers */
	for(int i = 0; i < network->num_layers; i += 1) {
		layer* curr_layer = &network->layers[i];

		/* Ignore if this isn't a recurrent layer */
		if(!curr_layer->recurrent) {
			continue;
		}

		/* Loop through all the neurons */
		for(int j = 0; j < curr_layer->num_neurons; j += 1) {

			/* Zero the recurrent_history */
			curr_layer->layer_neurons[j].recurrent_history = 0;
		}
	}
}

static void reset_derivatives(neural_network* network) {

	/* Reset the number of back propogations */
	network->num_back_propogations = 0;

	/* Loop through the network layers */
	for(int i = 1; i < network->num_layers; i += 1) {
		layer* curr_layer = &network->layers[i];

		/* Loop through all the neurons */
		for(int j = 0; j < curr_layer->num_neurons; j += 1) {


			/* Zero the weight derivatives */
			bzero(curr_layer->layer_neurons[j].weight_derivatives, network->layers[i-1].num_neurons * sizeof(double));

			/* Zero the bias derivative */
			curr_layer->layer_neurons[j].bias_derivative = 0;

			/* Zero the recurrent history derivative */
			curr_layer->layer_neurons[j].recurrent_weight_derivative = 0;

		}
	}
}



static void set_layer_outputs(double* inputs, size_t input_len, layer* layer) {
	info("[*] Network inputs: ");
	for(int i = 0; i < input_len; i += 1) {
		layer->layer_neurons[i].output = inputs[i];
		info("%f ", inputs[i]);
	}
		info("\n");
}

static double* get_layer_outputs(layer* layer) {

	/* Allocate an array for the outputs */
	double* ret = malloc(sizeof(double) * layer->num_neurons);
	if(!ret) {
		error("Failed to allocate output buffer.");
		return NULL;
	}

	/* Fill the array */
	for(int i = 0; i < layer->num_neurons; i += 1) {
		ret[i] = layer->layer_neurons[i].output;
	}

	/* Return the array */
	return ret;
}

static void update_history(neural_network* network) {

	/* Loop through the network layers */
	for(int i = 0; i < network->num_layers; i += 1) {
		layer* curr_layer = &network->layers[i];

		/* Ignore if this isn't a recurrent layer */
		if(!curr_layer->recurrent) {
			continue;
		}

		/* Loop through all the neurons */
		for(int j = 0; j < curr_layer->num_neurons; j += 1) {

			/* Update the recurrent_history */
			curr_layer->layer_neurons[j].recurrent_history = curr_layer->layer_neurons[j].output;
		}
	}
}

static void propogate_layer_forward(layer* input, layer* output) {

	/* Loop all the output neurons */
	for(int i = 0; i < output->num_neurons; i += 1) {

		/* Get the current neuron in our network */
		neuron* curr_neuron = &output->layer_neurons[i];

		/* Set weighted sum to our bias value */
		curr_neuron->weighted_sum = curr_neuron->bias;

		/* Add our recurrent layer if this is a recurrent layer */
		if(output->recurrent) {
			curr_neuron->weighted_sum += curr_neuron->recurrent_weight * curr_neuron->recurrent_history;
		}

		/* Loop through all the previous layers neurons */
		for(int j = 0; j < input->num_neurons; j += 1) {

			/* Add the weighted output to our sum */
			curr_neuron->weighted_sum += curr_neuron->weights[j] * input->layer_neurons[j].output;
		}

		/* Set our output based on the activation function */
		curr_neuron->output = activation_functions[curr_neuron->activation_index].activation(curr_neuron->weighted_sum);
	}
}

static void propogate_forward(neural_network* neural_net) {

	/* Propogate through all our network layers */
	for(int i = 0; i < (neural_net->num_layers-1); i += 1) {
		propogate_layer_forward(&neural_net->layers[i], &neural_net->layers[i+1]);
	}

	debug("[!] First output neuron: %f\n", neural_net->layers[neural_net->num_layers-1].layer_neurons[0].output);
#ifdef INFO
	info("[*] Output neurons: ");
	for(int i = 0; i < neural_net->layers[neural_net->num_layers-1].num_neurons; i += 1) {
		info("%f ", neural_net->layers[neural_net->num_layers-1].layer_neurons[i].output);

	}
	info("\n");
#endif
}

double* propogate_case_forward(neural_network* network, double* input, size_t input_len, size_t output_len) {


	/* Reset the networks history */
	reset_history(network);

	/* Allocate an output buffer */
	double* output = malloc(output_len * sizeof(double));
	if(!output) {
		error("Failed to allocate output buffer.");
		return NULL;
	}


	size_t input_offset = 0;
	size_t output_offset = 0;

	/* While the input hasn't been pushed through */
	while(input_len > 0) {

		/* Calculate the amount to add and the amount to output on this forward pass */
		size_t num_input_neurons = network->layers[0].num_neurons;
		size_t num_output_neurons = network->layers[0].num_neurons;

		size_t to_add = (input_len > num_input_neurons) ? num_input_neurons : input_len;
		size_t to_output = (output_len > num_output_neurons) ? num_output_neurons : output_len;

		/* Set out layer outputs */
		set_layer_outputs(&input[input_offset], to_add, &network->layers[0]);

		/* Propogate the network */
		propogate_forward(network);

		/* Get the output */
		double* iteration_outputs = get_layer_outputs(&network->layers[network->num_layers-1]);

		if(!iteration_outputs) {
			free(output);
			error("Failed to get iteration outputs.");
			return NULL;
		}

		/* Copy to the output buffer */
		memcpy(&output[output_offset], iteration_outputs, to_output * sizeof(double));

		/* Free the un-needed outputs */
		free(iteration_outputs);

		/* Propogate the network history */
		update_history(network);

		/* Update our variables */
		input_len -= to_add;
		input_offset += to_add;

		output_len -= to_output;
		output_offset += to_output;
	}

	return output;
}

static double cost(double value, double expected) {
	return pow(value - expected, 2);
}

static double cost_derivative(double value, double expected) {
	return 2 * (value - expected);
}

static double network_cost(neural_network* network, double* expected) {
	layer* output_layer = &network->layers[network->num_layers - 1];

	double ret = 0;

	/* Calculate the cost for the whole output layer */
	for(int i = 0; i < output_layer->num_neurons; i += 1) {
		ret += cost(output_layer->layer_neurons[i].output, expected[i]);
	}

	/* Average the cost */
	return ret / output_layer->num_neurons;
}

static void backpropogate_layer(neural_network* network, int layer_index, double* neuron_derivatives) {

	/* Base case: we don't need to propogate the input layer. */
	if(layer_index == 0) {
		return;
	}

	/* Get the current and previous network layer */
	layer* curr_layer = &network->layers[layer_index];
	layer* prev_layer = &network->layers[layer_index-1];

	/* Create and zero an array for the next layers derivatives */
	double* next_layer_derivatives = malloc(sizeof(double) * prev_layer->num_neurons);
	if(!next_layer_derivatives) {
		error("Failed to allocate next layer derivatives\n");
		return;
	}

	bzero(next_layer_derivatives, sizeof(double) * prev_layer->num_neurons);

	/* Loop through all the neurons in our layer */
	for(int i = 0; i < curr_layer->num_neurons; i += 1) {
		neuron* curr_neuron = &curr_layer->layer_neurons[i];

		/* Calculate the common derivative term - dCn/dz, where z is the weighted sum of our current neuron */
		double common_derivative_term = neuron_derivatives[i];
		common_derivative_term *= activation_functions[curr_neuron->activation_index].activation_derivative(curr_neuron->weighted_sum);

		/* Loop through the previous layers neurons */
		for(int j = 0; j < prev_layer->num_neurons; j += 1) {

			/* First calculate  dCn/dWj, which is just the common derivative term multiplied by the previous layers output */
			curr_neuron->weight_derivatives[j] += common_derivative_term * prev_layer->layer_neurons[j].output;

			/* Next, work out this neurons contribution to the previous neurons derivative dCn/dAj */
			next_layer_derivatives[j] += common_derivative_term * curr_neuron->weights[j];
		}

		/* Calculate the current neurons bias derivative - dCn/db */
		curr_neuron->bias_derivative += 1 * common_derivative_term;

		/* If this is a recurrent network calculate the derivative for the recurrent weight - dCn/dWh */
		if(curr_layer->recurrent) {
			curr_neuron->recurrent_weight_derivative += common_derivative_term * curr_neuron->recurrent_history;
		}

	}

	/* Propogate the previous layer and free our derivatives buffer */
	backpropogate_layer(network, layer_index - 1, next_layer_derivatives);
	free(next_layer_derivatives);

}


static void backpropogate_network(neural_network* network, double* expected_output) {
	
	/* To initialise the back propogation we need to create a neuron_derivatives array, which contains DCn/DA */
	layer* output_layer = &network->layers[network->num_layers - 1];
	
	double* input_derivatives = malloc(sizeof(double) * output_layer->num_neurons);
	if(!input_derivatives) {
		error("Failed to allocate input derivatives\n");
		return;
	}

	/* Get the derivative of the cost function with respect to the activation function for each output neuron */
	for(int i = 0; i < output_layer->num_neurons; i += 1) {
		input_derivatives[i] = cost_derivative(output_layer->layer_neurons[i].output, expected_output[i]);
	}


	/* Start backpropogation */
	backpropogate_layer(network, network->num_layers - 1, input_derivatives);

	/* Free our input derivatives */
	free(input_derivatives);

	/* Increment the number of back propogations */
	network->num_back_propogations += 1;

}

static void backpropogate_case(neural_network* network, test_case* test_case) {

	/* Reset the networks history */
	reset_history(network);

	size_t input_offset = 0;
	size_t output_offset = 0;

	size_t input_len = test_case->input_len;
	size_t output_len = test_case->output_len;


	/* While the input hasn't been pushed through */
	while(input_len > 0) {

		/* Calculate the amount to add and the amount to output on this forward pass */
		size_t num_input_neurons = network->layers[0].num_neurons;
		size_t num_output_neurons = network->layers[0].num_neurons;

		size_t to_add = (input_len > num_input_neurons) ? num_input_neurons : input_len;
		size_t to_output = (output_len > num_output_neurons) ? num_output_neurons : output_len;

		/* Set out layer outputs */
		set_layer_outputs(&test_case->input[input_offset], to_add, &network->layers[0]);

		/* Propogate the network forward */
		propogate_forward(network);

#ifdef INFO
		info("[*] Expected output: ");
		for(int i = 0; i < network->layers[network->num_layers-1].num_neurons; i += 1) {
			info("%f ", test_case->expected_output[output_offset + i]);
		}
			info("\n");
#endif
		debug("[!] Case Cost: %f\n", network_cost(network, &test_case->expected_output[output_offset]));

		/* Backpropogate */
		backpropogate_network(network, &test_case->expected_output[output_offset]);

		/* Propogate the network history */
		update_history(network);

		/* Update our variables */
		input_len -= to_add;
		input_offset += to_add;

		output_len -= to_output;
		output_offset += to_output;
	}
}

void backpropogate_cases(neural_network* network, test_case* cases, size_t num_cases, double learn_rate) {

	/* Reset the networks backpropogation variables */
	reset_derivatives(network);

	/* Backpropogate every case */
	for(int i = 0; i < num_cases; i += 1) {
		backpropogate_case(network, &cases[i]);
	}

	debug("[!] Number of back propogations steps: %d\n", network->num_back_propogations);

	/* Loop through each layer of the network, expect the input layer */
	for(int i = 1; i < network->num_layers; i += 1) {
		layer* curr_layer = &network->layers[i];

		/* Loop through each neuron of the network */
		for(int j = 0; j < curr_layer->num_neurons; j += 1) {
			neuron* curr_neuron = &curr_layer->layer_neurons[j];

			/* Loop through all the weights */
			for(int k = 0; k < network->layers[i-1].num_neurons; k += 1) {

				/* Nudge them by the negative of the average derivative, multiplied by the learn rate */
				curr_neuron->weights[k] -= (curr_neuron->weight_derivatives[k] / network->num_back_propogations) * learn_rate;
			}

			/* Nudge the bias and recurrent_weight by the negative of the average derivative, multiplied by the learn rate */
			curr_neuron->bias -= (curr_neuron->bias_derivative / network->num_back_propogations) * learn_rate;
			curr_neuron->recurrent_weight -= (curr_neuron->recurrent_weight_derivative / network->num_back_propogations) * learn_rate;

		}
	}

}
