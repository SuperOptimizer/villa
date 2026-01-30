#define CSVS_IMPLEMENTATION
#include "csvs.h"

/* This file exists solely to compile csvs.h into a shared library:
 *   gcc -shared -fPIC -O2 -o libcsvs.so csvs_shlib.c -llz4
 */
