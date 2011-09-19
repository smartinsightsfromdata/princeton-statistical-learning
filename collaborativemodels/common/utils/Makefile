
SOURCE_FILES=logging.cpp \
	           math_utils.cpp \
		         strings.cpp
OBJECT_FILES=$(patsubst %.cpp, %.o,$(SOURCE_FILES))


GSL_CFLAGS=$(shell gsl-config --cflags)
CPPFLAGS=-Wall -O3 $(GSL_CFLAGS)

libutils.a(*.o): $(OBJECT_FILES)
	ar cr $@ $?
