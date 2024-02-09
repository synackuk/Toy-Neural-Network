#ifndef COMMON_H
#define COMMON_H

#include <includes/test_case.h>
#include <includes/nn.h>

double* buf_to_bits(char* buf, size_t* out_size, size_t buf_len);
size_t get_file_size(char* filename);
int read_file(char* filename, char** out_buf, size_t* file_len);


#endif
