#include <includes/common.h>
#include <unistd.h>

static size_t count_string_tokens(char* input, char token) {
	size_t ret = 0;
	int i = 0;

	/* Loop through the string and count the number of a given token */
	while(input[i] != '\0') {
		if(input[i] == token) {
			ret += 1;
		}
		i += 1;
	}
	return ret;
}


static neural_network* gen_nn_from_params(char* in_string, bool recursive) {

	/* Get the number of layers and validate the correctness of this */
	size_t num_layers = count_string_tokens(in_string, ',') + 1;
	if(num_layers < 2) {
		error("Not enough layers for neural network\n");
		return NULL;
	}

	size_t* layer_sizes = malloc(sizeof(size_t) * num_layers);
	if(!layer_sizes) {
		error("Failed to allocate space for layer sizes array\n");
		free(layer_sizes);
		return NULL;
	}


	/* Loop through the string by commas */
	char* token = strtok(in_string, ",");

	int i = 0;
	while (token != NULL) {

		/* Get each layer size */
		layer_sizes[i] = atoi(token);
		if(layer_sizes[i] <= 0) {
			error("Invalid layer size for neural network\n");
			free(layer_sizes);
			return NULL;
		}

		i += 1;
		token = strtok(NULL, ",");
	}

	/* Allocate the recursive array and populate it */
	bool* recursive_array = malloc(sizeof(bool) * num_layers);
	if(!recursive_array) {
		error("Failed to allocate space for recursive array\n");
		free(layer_sizes);
		return NULL;
	}

	for(int i = 0; i < num_layers; i += 1) {
		if(i > 0 && i < (num_layers-1) && recursive) {
			recursive_array[i] = true;
			continue;
		}
		recursive_array[i] = false;
	}

	/* Initialise a new neural net */
	neural_network* ret = init_neural_network(recursive_array, layer_sizes, num_layers);
	free(layer_sizes);
	free(recursive_array);

	return ret;
}


static void generate_training_data_from_input(char* input_string) {

	/* Get the number of input cases */
	size_t num_cases = count_string_tokens(input_string, ',') + 1;
	char** input_filenames = malloc(sizeof(char*) * num_cases);
	if(!input_filenames) {
		error("Failed to allocate input filenames array\n");
		goto out;
	}

	char** output_filenames = malloc(sizeof(char*) * num_cases);
	if(!output_filenames) {
		error("Failed to allocate output filenames array\n");
		goto out;
	}

	char* token = strtok(input_string, ",");

	int i = 0;
	while (token != NULL) {

		/* Get the input filename */
		input_filenames[i] = strdup(token);
		if(!input_filenames[i]) {
			error("Failed to allocate output filenames array\n");
			goto out;
		}

		/* Find the equals sign in our input filename */
		int equals_offset = 0;
		int j = 0;
		while(input_filenames[i][j] != '\0') {
			if(input_filenames[i][j] == '=') {
				equals_offset = j;
				break;
			}
			j += 1;
		}

		if(equals_offset == 0) {
			error("Data isn't formatted as input=expected_output\n");
			goto out;
		}

		/* The output filename starts after the equals sign, and the input filename is before it */
		output_filenames[i] = strdup(&token[equals_offset+1]);
		input_filenames[i][equals_offset] = '\0';

		i += 1;
		token = strtok(NULL, ",");
	}


	export_training_data(input_filenames, output_filenames, "data.td", num_cases);

out:
	if(input_filenames) {
		for(int j = 0; j < i; j += 1) {
			free(input_filenames[j]);
		}
		free(input_filenames);
	}

	if(output_filenames) {
		for(int j = 0; j < i; j += 1) {
			free(output_filenames[j]);
		}
		free(output_filenames);
	}
}

static void print_usage(char** argv) {
	printf("Usage: %s [options]\n", argv[0]);
	printf("\t-h\tPrint the help dialog\n");
	printf("\t-l <network>\tLoad a network from a file\n");
	printf("\t-n <layer_sizes>\tCreate a new network - layer sizes should be comma deliminated\n");
	printf("\t-r <layer_sizes>\tCreate a new recurrent network - layer sizes should be comma deliminated\n");
	printf("\t-s <filepath>\tSave the network to a file\n");
	printf("\t-f <input>\tLoad input data from a file\n");
	printf("\t-e <output_length>\tLength of output data\n");
	printf("\t-t <test_cases>\tTrain the network using test cases from a file\n");
	printf("\t-o <output>\tSave network output to a file\n");
	printf("\t-g <case_files>\tGenerate test cases from files, in the form input=output, input=output - saved as data.td\n");
	printf("\t-a <learn_rate>\tSet a custom learning rate for back propogation default (0.05)\n");
	printf("\t-i <num_iterations>\tSet a custom number of back propogation iterations (default 100)\n");

}

int main(int argc, char** argv) {
	srand(time(NULL));

	int ret = 0;
	
	/* Prepare all our variables */
	neural_network* network = NULL;
	char* network_out_file = NULL;
	char* input_data_file = NULL;
	char* output_data_file = NULL;

	test_case* training_data = NULL;
	size_t num_test_cases = 0;

	size_t output_len = 0;

	int num_iterations = 100;
	double learn_rate = 0.05;

	double* output = NULL;

	/* Read in all the options */
	int opt;
	while ((opt = getopt(argc, argv, "l:n:r:s:e:f:t:o:a:i:g:h")) != -1) {
		switch(opt) {
		case 'l':
			if(network) {
				error("You cannot load multiple networks at once\n");
				return 0;
			}
			network = import_neural_network(optarg);
			if(!network) {
				return 0;
			}
			break;
		case 'n':
			if(network) {
				error("You cannot load multiple networks at once\n");
				return 0;
			}
			network = gen_nn_from_params(optarg, false);
			if(!network) {
				return 0;
			}
			break;
		case 'r':
			if(network) {
				error("You cannot load multiple networks at once\n");
				return 0;
			}
			network = gen_nn_from_params(optarg, true);
			if(!network) {
				return 0;
			}
			break;
		case 's':
			network_out_file = optarg;
			break;
		case 'e':
			output_len = atoi(optarg) * 8;
			if(output_len == 0) {
				error("Output length incorrect\n");
				return 0;
			}
			break;
		case 'f':
			input_data_file = optarg;
			break;
		case 't':
			ret = import_training_data(optarg, &training_data, &num_test_cases);
			if(ret != 0) {
				return 0;
			}
			break;
		case 'o':
			output_data_file = optarg;
			break;
		case 'a':
			learn_rate = atof(optarg);
			break;
		case 'i':
			num_iterations = atoi(optarg);
			break;
		case 'g':
			generate_training_data_from_input(optarg);
			return 0;
		case 'h':
		default:
			print_usage(argv);
			return 0;
		}
	}
	if(!network) {
		error("No neural network loaded\n");
		return 0;
	}

	/* If we have training data, train the network */
	if(training_data) {
		for(int i = 0; i < num_iterations; i += 1) {
			backpropogate_cases(network, training_data, num_test_cases, learn_rate);
		}
	}

	/* If we have an input file read it in and propogate it */
	if(input_data_file) {
		if(output_len == 0) {
			free_neural_network(network);
			error("Output length unspecified\n");
			return 0;
		}

		char* input_data;
		size_t input_data_len;
		ret = read_file(input_data_file, &input_data, &input_data_len);
		if(ret != 0) {
			free_neural_network(network);
			return 0;
		}

		size_t input_len;
		double* input = buf_to_bits(input_data, &input_len, input_data_len);
		free(input_data);
		if(!input) {
			free_neural_network(network);
			return 0;
		}

		input_len /= sizeof(double);
		output = propogate_case_forward(network, input, input_len, output_len);
	}

	/* If we have somewhere to save the output, save it */
	if(output_data_file && output) {
		FILE* f = fopen(output_data_file, "wb");
		if(!f) {
			error("Failed to open output file\n");
			free(output);
			free_neural_network(network);
			return 0;
		}
		char byte = 0;
		for(int i = 0; i < output_len; i += 1) {
			byte |= ((int)output[i]) << (7 - (i % 8));
			if(i % 8 == 0) {
				fwrite(&byte, sizeof(char), 1, f);
				byte = 0;
			}
		}
		fclose(f);
	}

	/* If we should export the network, do that */
	if(network_out_file) {
		export_neural_network(network, network_out_file);
	}

	if(output) {
		free(output);
	}

	free_neural_network(network);
	return 0;
}
