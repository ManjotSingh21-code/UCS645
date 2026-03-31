#include <iostream>
#include <omp.h>

int main(int argc, char *argv[]) {

    int N = 1500;
    int steps = 1500;
    int threads = 1;

    if (argc > 1)
        threads = std::atoi(argv[1]);

    omp_set_num_threads(threads);

    double** T = new double*[N];
    double** Tnew = new double*[N];

    for (int i = 0; i < N; i++) {
        T[i] = new double[N]();
        Tnew[i] = new double[N]();
    }

    for (int j = 0; j < N; j++)
        T[0][j] = 100.0;

    double start = omp_get_wtime();

    for (int t = 0; t < steps; t++) {

        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {

                Tnew[i][j] = 0.25 * (
                    T[i+1][j] +
                    T[i-1][j] +
                    T[i][j+1] +
                    T[i][j-1]
                );
            }
        }

        double** temp = T;
        T = Tnew;
        Tnew = temp;
    }

    double end = omp_get_wtime();

    std::cout << "Threads Used: " << threads << std::endl;
    std::cout << "Execution Time: " << (end - start) << " seconds" << std::endl;

    for (int i = 0; i < N; i++) {
        delete[] T[i];
        delete[] Tnew[i];
    }

    delete[] T;
    delete[] Tnew;

    return 0;
}
