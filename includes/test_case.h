#ifndef TEST_CASE_H
#define TEST_CASE_H

#include <stdlib.h>
#include <stdio.h>

#define TRAINING_DATA_MAGIC 0x5453554B /* SUKT */

/* Generic test case */
typedef struct {
	double* input;
	double* expected_output;
	size_t input_len;
	size_t output_len;
} test_case;


typedef struct {
	size_t input_len;
	size_t output_len;
} file_test_case;

typedef struct {
	uint32_t magic; /* SUKT */
	size_t num_test_cases;
} training_data_header;

void test_case_free(test_case case_to_free);
void test_cases_free(test_case* cases_to_free, size_t num_cases);
int export_training_data(char** input_filenames, char** expected_output_filenames, char* output_filename, size_t num_cases);
int import_training_data(char* filename, test_case** ret_cases, size_t* num_cases);

#endif
