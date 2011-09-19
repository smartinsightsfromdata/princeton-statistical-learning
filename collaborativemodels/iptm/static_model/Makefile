# Copyright 2010 Princeton University.
# All Rights Reserved.
#
# Author: Sean M. Gerrish (sgerrish@cs.princeton.edu)
#

CC=g++

COMMON_PATH=../../common/utils

GSL_CFLAGS=$(shell gsl-config --cflags)
GSL_LIBS=$(shell gsl-config --libs)

CFLAGS=-Wall -O3 $(GSL_CFLAGS) -I$(COMMON_PATH)

LDFLAGS=-lm $(GSL_LIBS) -L$(COMMON_PATH) -lutils -lgflags

SOURCE_FILES=main.c \
	           corpus.c \
		       	 flags.c \
		         infer.c \
			       gsl-wrappers.c

all: legis

legis:	$(SOURCE_FILES)
	$(CC) $(CFLAGS) $? -o $@ $(LDFLAGS)

clean:
	rm -f *.o
