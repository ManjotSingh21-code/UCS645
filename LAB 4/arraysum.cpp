#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate
#include <mpi.h>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 100;
    
    // Ensure the workload is divisible by the number of processes
    if (N % size != 0) {
        if (rank == 0) {
            std::cerr << "Error: N (" << N << ") must be divisible by size (" << size << ")" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int chunk_size = N / size;
    std::vector<int> data;
    std::vector<int> local_data(chunk_size);

    // Only the root process needs to allocate and initialize the full dataset
    if (rank == 0) {
        data.resize(N);
        std::iota(data.begin(), data.end(), 1); // Fills 1, 2, 3... N
    }

    // Synchronization and Timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Distribute the data
    MPI_Scatter(
        data.data(), chunk_size, MPI_INT, 
        local_data.data(), chunk_size, MPI_INT, 
        0, MPI_COMM_WORLD
    );

    // Calculate local sum using C++ STL
    int local_sum = std::accumulate(local_data.begin(), local_data.end(), 0);

    // Reduce local sums into a global sum
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Output results from the root process
    if (rank == 0) {
        std::cout << "Global Sum = " << global_sum << " (Expected = 5050)" << std::endl;
        std::cout << "Execution Time: " << (end_time - start_time) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
