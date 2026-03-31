#include <iostream>
#include <vector>
#include <numeric> 
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    const int N = 8;
    int A[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[N] = {8, 7, 6, 5, 4, 3, 2, 1};

    
    if (N % size != 0) {
        if (rank == 0) std::cerr << "Error: N must be divisible by the number of processes." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int chunk = N / size;
    std::vector<int> localA(chunk);
    std::vector<int> localB(chunk);

    
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    
    MPI_Scatter(A, chunk, MPI_INT, localA.data(), chunk, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, chunk, MPI_INT, localB.data(), chunk, MPI_INT, 0, MPI_COMM_WORLD);

    
    int local_dot = std::inner_product(localA.begin(), localA.end(), localB.begin(), 0);

     
    int global_dot = 0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Dot Product = " << global_dot << " (Expected = 120)" << std::endl;
        std::cout << "Time: " << (end - start) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
