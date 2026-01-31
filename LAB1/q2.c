#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    int N = 100;
    int X[N][N], Y[N][N], Z[N][N];

    // initialise with random values
    srand(0);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            X[i][j] = rand() % 5; // range= [0-4]
            Y[i][j] = rand() % 5;
        }
    }

    double s_time, e_time;
    s_time = omp_get_wtime();
    int j = 0, k = 0;
#pragma omp parallel for private(j, k)
    for (int i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                Z[i][j] += X[i][k] + Y[k][j];
            }
        }
    }

    e_time = omp_get_wtime();
    printf("Execution time: %f\n", e_time - s_time);
    return 0;
}
