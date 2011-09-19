#ifndef __UTIL_MATH_UTILS_H__
#define __UTIL_MATH_UTILS_H__

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace slap_utils {

const double PI=3.141592654;

double UniformRNG();
double log_sum(double log_a, double log_b);
double trigamma(double x);
double digamma(double x);
double log_gamma(double x);
void make_directory(char* name);
int argmax(double* x, int n);

}  // namespace slap_utils

#endif  // __UTIL_MATH_UTILS_H__
