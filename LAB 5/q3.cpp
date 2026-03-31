#include <iostream>
#include <iomanip> 
#include <mpi.h>

// Using a large N to demonstrate parallel workload
const long N = 500000000;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double multiplier = 0.0;

    
    if (rank == 0) {
        multiplier = 2.0;
    }

    // Broadcast the multiplier from rank 0 to all other processes
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    long chunk = N / size;
    double local_sum = 0.0;

    // Synchronize before timing the computation
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    
    for (long i = 0; i < chunk; ++i) {
        double A = 1.0;
        double B = 2.0 * multiplier;
        local_sum += A * B;
    }

    double global_sum = 0.0;
    // Reduce all local_sums into global_sum on rank 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        // Using fixed and setprecision to handle large double outputs nicely
        std::cout << "Dot Product Result: " << std::fixed << std::setprecision(2) << global_sum << std::endl;
        std::cout << "Computation Time: " << (end_time - start_time) << " seconds" << std::endl;
        
        // Validation check
        double expected = (double)N * 4.0;
        std::cout << "Expected Result:   " << expected << std::endl;
    }

    MPI_Finalize();
    return 0;
}
