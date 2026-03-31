#include <iostream>
#include <cmath>
#include <omp.h>

int main(int argc, char *argv[]) {

    int threads = 0;

    if (argc > 1) {
        threads = atoi(argv[1]);
        omp_set_num_threads(threads);
    }

    if (threads == 0) {
        threads = omp_get_max_threads();
    }

    int N = 1000;
    double epsilon = 1.0, sigma = 1.0;
    double rc2 = 2.5 * 2.5;

    double* x = new double[N];
    double* y = new double[N];
    double* z = new double[N];

    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        z[i] = drand48();
    }

    double potential = 0.0;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:potential)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {

            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];

            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < 1e-4 || r2 > rc2) continue;

            double inv2  = (sigma * sigma) / r2;
            double inv6  = inv2 * inv2 * inv2;
            double inv12 = inv6 * inv6;

            potential += 4 * epsilon * (inv12 - inv6);
        }
    }

    double end = omp_get_wtime();

    std::cout << "Threads Used: " << threads << std::endl;
    std::cout << "Execution Time: " << (end - start) << " seconds" << std::endl;
    std::cout << "Total Potential Energy: " << potential << std::endl;

    delete[] x;
    delete[] y;
    delete[] z;

    return 0;
}
