#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    int N = 65536;
    double *X = (double *)malloc(N * sizeof(double));
    double *Y = (double *)malloc(N * sizeof(double));


    //initialise with random values
    srand(0);
    for (int i = 0; i < N; i++)
    {
        X[i] = (double)rand() / RAND_MAX;
        Y[i] = (double)rand() / RAND_MAX;
    }
    //scalar
    double a = 3.12;

    double s_time, e_time;
    s_time = omp_get_wtime();

    // #threads
    //omp_set_num_threads(4);

    #pragma omp parallel
    for (int i = 0; i < N; i++)
    {
        X[i] = a * X[i] + Y[i];
    }

    e_time = omp_get_wtime();
    printf("Execution time: %f\n", e_time - s_time);
    return 0;
}
