all : main.o tensor.o baseline.o shared_mem.o
	nvcc -g -G main.o tensor.o baseline.o shared_mem.o -o forward

main.o : main.cu
	nvcc -g -G -std=c++11 main.cu -c -o main.o

tensor.o : tensor.cu
	nvcc -g -G -std=c++11 tensor.cu -c -o tensor.o

baseline.o : baseline.cu
	nvcc -g -G -std=c++11 baseline.cu -c -o baseline.o

shared_mem.o : shared_mem.cu
	nvcc -g -G -std=c++11 shared_mem.cu -c -o shared_mem.o

clean:
	rm -f *.o forward
