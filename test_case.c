#include <includes/common.h>

void test_case_free(test_case case_to_free) {
	free(case_to_free.input);
	free(case_to_free.expected_output);
}


void test_cases_free(test_case* cases_to_free, size_t num_cases) {
	for(int i = 0; i < num_cases; i += 1) {
		test_case_free(cases_to_free[i]);
	}
}


int export_training_data(char** input_filenames, char** expected_output_filenames, char* output_filename, size_t num_cases) {

	/* First open the output file */
	FILE* output_file = fopen(output_filename, "wb");

	if(!output_file) {
		error("Failed to open output file\n");
		return -1;
	}

	/* Write in our file header */
	training_data_header header = {.magic=TRAINING_DATA_MAGIC, .num_test_cases=num_cases};

	fwrite(&header, sizeof(training_data_header), 1, output_file);


	/* Next write the test case headers */
	for(int i = 0; i < num_cases; i += 1) {
		file_test_case test_case;

		test_case.input_len = get_file_size(input_filenames[i]) * 8;
		if(!test_case.input_len) {
			fclose(output_file);
			return -1;
		}

		test_case.output_len = get_file_size(expected_output_filenames[i]) * 8;
		if(!test_case.output_len) {
			fclose(output_file);
			return -1;
		}

		fwrite(&test_case, sizeof(file_test_case), 1, output_file);

	}

	/* Finally write in the input and output data */
	for(int i = 0; i < num_cases; i += 1) {
		char* file_buf;
		size_t file_len;
		size_t bit_len;
		char* filenames[] = {input_filenames[i], expected_output_filenames[i]};

		for(int j = 0; j < 2; j += 1) {
			/* Read in our file */
			int ret = read_file(filenames[j], &file_buf, &file_len);
			if(ret != 0) {
				fclose(output_file);
				return -1;
			}


			/* Convert to bits */
			double* bit_buf = buf_to_bits(file_buf, &bit_len, file_len);
			free(file_buf);
			if(!bit_buf) {
				fclose(output_file);
				return -1;
			}
	
			/* Write it out */
			fwrite(bit_buf, bit_len, 1, output_file);

			free(bit_buf);
		}

	}

	fclose(output_file);
	return 0;
}


int import_training_data(char* filename, test_case** ret_cases, size_t* num_cases) {
	size_t file_length;
	char* file_buf;

	int ret = read_file(filename, &file_buf, &file_length);
	if(ret != 0) {
		return -1;
	}

	/* Verify we can fit a file header at least */
	if(file_length < sizeof(training_data_header)) {
		error("File too small\n");
		free(file_buf);
		return -1;

	}


	training_data_header* header = (training_data_header*)file_buf;

	/* Verify file magic */
	if(header->magic != TRAINING_DATA_MAGIC) {
		error("Bad magic file\n");		
		free(file_buf);
		return -1;
	}

	/* Calculate the end of the file */
	size_t file_end = sizeof(training_data_header) + sizeof(file_test_case) * header->num_test_cases;


	if(file_length < file_end) {
		error("File too small\n");		
		free(file_buf);
		return -1;
	}

	file_test_case* test_case_info = (file_test_case*)&file_buf[sizeof(training_data_header)];

	/* Update the end of the file */
	for(int i = 0; i < header->num_test_cases; i += 1) {
		file_end += test_case_info[i].input_len * sizeof(double);
		file_end += test_case_info[i].output_len * sizeof(double);
	}

	/* Verify we're safe */
	if(file_length < file_end) {
		error("File too small\n");		
		free(file_buf);
		return -1;
	}

	/* Allocate our test cases */
	*ret_cases = malloc(header->num_test_cases * sizeof(test_case));
	if(!*ret_cases) {
		error("Failed to allocate test case buffer\n");		
		free(file_buf);
		return -1;
	}
	*num_cases = header->num_test_cases;
	
	size_t file_offset = sizeof(training_data_header) + sizeof(file_test_case) * header->num_test_cases;

	for(int i = 0; i < header->num_test_cases; i += 1) {

		/* Set the current cases input and output length */
		test_case* curr_case = &(*ret_cases[i]);

		curr_case->input_len = test_case_info[i].input_len;
		curr_case->output_len = test_case_info[i].output_len;

		/* Allocate our input */
		curr_case->input = malloc(curr_case->input_len * sizeof(double));
		if(!curr_case->input) {
			error("Failed to allocate test case buffer\n");		
			test_cases_free(*ret_cases, i);
			free(file_buf);
			return -1;
		}

		/* Allocate our expected output */
		curr_case->expected_output = malloc(curr_case->output_len * sizeof(double));
		if(!curr_case->expected_output) {
			error("Failed to allocate test case buffer\n");
			free(curr_case->input);	
			test_cases_free(*ret_cases, i);
			free(file_buf);
			return -1;
		}

		/* Copy our input and expected output into place */
		memcpy(curr_case->input, &file_buf[file_offset], test_case_info[i].input_len * sizeof(double));
		file_offset += test_case_info[i].input_len * sizeof(double);

		memcpy(curr_case->expected_output, &file_buf[file_offset], test_case_info[i].output_len * sizeof(double));
		file_offset += test_case_info[i].output_len * sizeof(double);

#ifdef INFO
		printf("[*] Case input: ");
		for(int j = 0; j < curr_case->input_len; j += 1) {
			printf("%f ", curr_case->input[j]);
		}
		printf("\n");

		printf("[*] Case output: ");
		for(int j = 0; j < curr_case->output_len; j += 1) {
			printf("%f ", curr_case->expected_output[j]);
		}
		printf("\n");

#endif

	}


	free(file_buf);
	return 0;
}
