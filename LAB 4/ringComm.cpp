#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = 0;

    // Only the root starts with the initial value
    if (rank == 0) {
        value = 100;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Ring topology communication loop
    for (int step = 0; step < size; ++step) {
        // The sender for this specific step
        if (rank == step) {
            int next = (rank + 1) % size;
            std::cout << "Process " << rank << " sending " << value 
                      << " to process " << next << std::endl;
            
            MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        }
        
        // The receiver for this specific step
        if (rank == (step + 1) % size) {
            int prev = (rank - 1 + size) % size;
            
            MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            value += rank; // Modify the value
            
            std::cout << "Process " << rank << " received value " << value << std::endl;
        }

        // Ensure each step of the ring completes before moving to the next
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Final value returned to root: " << value << std::endl;
        std::cout << "Total execution time: " << (end_time - start_time) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
