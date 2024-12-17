#ifndef COMMON_H
#define COMMON_H

#define my_decimal double

// Value that is safe to square without underflow
#define MY_DECIMAL_UNDERFLOW_LIMIT 1e-18

#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

// Returns a my_decimal in [0, 1)
my_decimal random_my_decimal() {
    return ((my_decimal) rand()) / ((my_decimal) RAND_MAX + 1);
}

// Returns a my_decimal in [min, max)
my_decimal random_my_decimal_in(my_decimal min, my_decimal max) {;
    return min + (max-min)*random_my_decimal();
}

#endif