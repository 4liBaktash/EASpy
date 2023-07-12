#! /bin/bash

rm libEASpy.so
rm tools_EASpy.o

gcc -fPIC -O2 -fopenmp -flto -mtune=native -march=znver1 -c -Wall -lm -lgsl -lgslcblas tools_EASpy.c
gcc -O2 -fopenmp -flto -mtune=native -march=znver1 -shared -Wl,-soname,libEASpy.so -o libEASpy.so tools_EASpy.o -lm -lgsl -lgslcblas    
