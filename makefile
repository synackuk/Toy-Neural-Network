all:
	@gcc -o nn nn.c test_case.c common.c main.c -O3 -Wall -pedantic -O3 -std=gnu2x -I.

clean:
	@$(RM) -rf nn