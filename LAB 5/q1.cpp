#include <iostream>
#include <vector>
#include <mpi.h>

const int N = 65536;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 2.5;

    // Standard C++ way to handle divisibility and chunking
    if (N % size != 0 && rank == 0) {
        std::cerr << "Warning: N is not perfectly divisible by the number of processes." << std::endl;
    }

    int chunk = N / size;

    // Use std::vector for automatic memory management (RAII)
    std::vector<double> X(chunk, 1.0);
    std::vector<double> Y(chunk, 2.0);

    // Synchronize processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // The SAXPY Kernel
    for (int i = 0; i < chunk; ++i) {
        X[i] = a * X[i] + Y[i];
    }

    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "MPI Time: " << (end - start) << " seconds" << std::endl;
        // Result check: 2.5 * 1.0 + 2.0 = 4.5
        std::cout << "Sample Result (X[0]): " << X[0] << std::endl;
    }

    // No free() needed; vectors are destroyed when they go out of scope
    MPI_Finalize();
    return 0;
}
