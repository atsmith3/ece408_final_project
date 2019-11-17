all : main.o
	nvcc -g -G main.o -o forward

main.o : main.cu
	nvcc -g -G main.cu -c -o main.o

clean:
	rm -f *.o forward
