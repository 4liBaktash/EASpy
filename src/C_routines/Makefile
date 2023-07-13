CC = gcc
CFLAGS = -fPIC -O2 -fopenmp -flto -mtune=native -march=znver1 -c -Wall -lm -lgsl -lgslcblas
LDFLAGS = -shared -O2 -fopenmp -flto -mtune=native -march=znver1
RM = rm -f
TARGET = libEASpy.so

SRCS = tools_EASpy.c
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET}

$(TARGET): $(OBJS)
	$(CC) ${LDFLAGS} -Wl,-soname,$(TARGET) -o $@ $^ -lm -lgsl -lgslcblas

.PHONY: clean
clean:
	-${RM} ${OBJS} $(TARGET)
