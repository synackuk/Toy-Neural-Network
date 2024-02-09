#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <strings.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <includes/test_case.h>

#define DEBUG
#define INFO

/* Allow for debug and info logging */

#ifdef DEBUG
#define debug printf
#else
#define debug(...) 
#endif

#ifdef INFO
#define info printf
#else
#define info(...) 
#endif

#define NEURAL_NETWORK_MAGIC 0x4E4E5553 /* SUNN */

#define error(s) printf("Error on line %d: %s\n", __LINE__, s)

/* Get a random number between 0 and 1 */
#define uniform_decimal() ((double)rand()/(double)(RAND_MAX))

/* Generic activation function */
typedef double (*activation_func)(double input);

typedef struct {
	activation_func activation;
	activation_func activation_derivative;
} activation_function;

/* Generic neuron */
typedef struct {
	uint32_t activation_index;
	double* weights;
	double bias;
	double weighted_sum;
	double output;
	double recurrent_weight;
	double recurrent_history;
	double* weight_derivatives;
	double bias_derivative;
	double recurrent_weight_derivative;
} neuron;


/* Generic neural net layer */
typedef struct {
	neuron* layer_neurons;
	size_t num_neurons;
	bool recurrent;
} layer;

/* Generic neural net */
typedef struct {
	layer* layers;
	size_t num_layers;
	int num_back_propogations;
} neural_network;


typedef struct {
	double bias;
	double recurrent_weight;
	uint32_t activation_index;
	size_t num_weights;
	size_t neuron_len;
} file_neuron;

typedef struct {
	size_t num_neurons;
	size_t layer_len;
	bool recurrent;
} file_layer;


typedef struct {
	uint32_t magic; /* 'SUNN' */
	size_t num_layers;
} neural_network_file_header;

neural_network* init_neural_network(bool* recurrent_layer, size_t* layer_sizes, size_t num_layers);
void free_neural_network(neural_network* network);

neural_network* import_neural_network(char* filename);
void export_neural_network(neural_network* network, char* filename);

double* propogate_case_forward(neural_network* network, double* input, size_t input_len, size_t output_len);

void backpropogate_cases(neural_network* network, test_case* cases, size_t num_cases, double learn_rate);

#endif
