#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Modern C++ Random Number Generation
    std::mt19937 gen(1337 + rank); // Seeded with rank
    std::uniform_int_distribution<> dis(0, 999);

    const int count = 10;
    std::vector<int> local_values(count);
    for (int& val : local_values) {
        val = dis(gen);
    }

    // Find local max and min using STL algorithms
    auto [min_it, max_it] = std::minmax_element(local_values.begin(), local_values.end());
    
    // Structs for MPI_2INT
    // Note: MPI_2INT specifically expects two contiguous ints
    struct DataPair {
        int value;
        int rank;
    };

    DataPair in_max { *max_it, rank };
    DataPair in_min { *min_it, rank };
    DataPair out_max, out_min;

    // Synchronization and Timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Use MPI_2INT to reduce both the value and the rank (index)
    MPI_Reduce(&in_max, &out_max, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&in_min, &out_min, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Global Max = " << out_max.value << " from process " << out_max.rank << std::endl;
        std::cout << "Global Min = " << out_min.value << " from process " << out_min.rank << std::endl;
        std::cout << "Time: " << (end - start) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
