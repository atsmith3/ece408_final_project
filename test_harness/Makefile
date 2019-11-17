all : main.o tensor.o baseline.o
	nvcc -g -G main.o tensor.o baseline.o -o forward

main.o : main.cu
	nvcc -g -G main.cu -c -o main.o

tensor.o : tensor.cu
	nvcc -g -G tensor.cu -c -o tensor.o

baseline.o : baseline.cu
	nvcc -g -G baseline.cu -c -o baseline.o

clean:
	rm -f *.o forward
