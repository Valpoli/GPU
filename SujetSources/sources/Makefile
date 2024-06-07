
all: main1 main2 main3

main1: main1.cu exercice.cu exercice.h image.o
	nvcc main1.cu exercice.cu image.o -o main1
main2: main2.cu exercice.cu exercice.h image.o
	nvcc main2.cu exercice.cu image.o -o main2
main3: main3.cu exercice.cu exercice.h image.o
	nvcc main3.cu exercice.cu image.o -o main3

image.o: image.cpp image.h
	g++ -c image.cpp

.PHONY: clean
clean:
	rm image.o main1 main2 main3