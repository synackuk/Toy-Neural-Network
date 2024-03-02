#include <includes/common.h>

double* buf_to_bits(char* buf, size_t* out_size, size_t buf_len) {
	
	/* The output length is 8 times the input length, with each bit taking up a double */
	*out_size = buf_len * 8 * sizeof(double);
	double* ret_array = malloc(*out_size);
	if(!ret_array) {
		error("Failed to allocate bit array\n");
		return NULL;
	}

	size_t curr_ret_offset = 0;
	for(int i = 0; i < buf_len; i += 1) {
		/* Get the byte we're interested in */
		char curr_byte = buf[i];
		for(int bit = 7; bit >= 0; bit -= 1) {
			/* Set that bit */
			ret_array[curr_ret_offset] = (double)((curr_byte >> bit) & 1);
			curr_ret_offset += 1;
		}
	}
	return ret_array;
}


size_t get_file_size(char* filename) {
	/* Open the file */
	FILE* f = fopen(filename, "rb");
	if(!f) {
		error("Failed to open file\n");
		fclose(f);
		return 0;
	}

	/* Get the length */
	fseek(f, 0, SEEK_END);
	size_t ret = ftell(f);
	fseek(f, 0, SEEK_SET);

	fclose(f);
	return ret;
}

int read_file(char* filename, char** out_buf, size_t* file_len) {
	/* Open the file */
	FILE* f = fopen(filename, "rb");
	if(!f) {
		error("Failed to open file\n");
		fclose(f);
		return -1;
	}
	
	/* Get the file length */
	fseek(f, 0, SEEK_END);
	*file_len = ftell(f);
	fseek(f, 0, SEEK_SET);

	/* Allocate space */
	*out_buf = malloc(*file_len);
	if(!*out_buf) {
		error("Failed to allocate file buffer\n");
		fclose(f);
		return -1;
	}

	/* Read it in */
	fread(*out_buf, 1, *file_len, f);
	fclose(f);
	return 0;
}
