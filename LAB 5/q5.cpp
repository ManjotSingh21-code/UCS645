#include <iostream>
#include <vector>
#include <mpi.h>


bool isPerfect(int n) {
    if (n < 2) return false;
    int sum = 1; 
    for (int i = 2; i <= n / 2; ++i) {
        if (n % i == 0) {
            sum += i;
        }
    }
    return sum == n;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int MAX_VAL = 10000;
    const int STOP_SIGNAL = -1;

    
    if (size < 2) {
        if (rank == 0) {
            std::cerr << "Error: This program requires at least 2 processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        
        int next_num = 2;
        int active_workers = size - 1;

        while (active_workers > 0) {
            int worker_ready_signal;
            MPI_Status status;

            // Wait for any worker to signal they are ready for work
            MPI_Recv(&worker_ready_signal, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            if (next_num <= MAX_VAL) {
                // Send the next task to the specific worker that just messaged us
                MPI_Send(&next_num, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                next_num++;
            } else {
                // No more numbers to check; tell the worker to shut down
                MPI_Send(&STOP_SIGNAL, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                active_workers--;
            }
        }
        std::cout << "Master process: Search complete." << std::endl;

    } else {
        
        int num_to_check = 0; 

        while (true) {
            // Tell the master we are ready and wait for a number
            MPI_Send(&num_to_check, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&num_to_check, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            
            if (num_to_check == STOP_SIGNAL) break;

            if (isPerfect(num_to_check)) {
                // Use std::flush or std::endl to ensure output appears immediately
                std::cout << "Process " << rank << " found Perfect Number: " << num_to_check << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
