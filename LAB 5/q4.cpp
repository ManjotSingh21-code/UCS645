#include <iostream>
#include <cmath>
#include <mpi.h>

// Helper function using C++ bool type
bool isPrime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i <= std::sqrt(n); i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int MAX_VAL = 100;
    const int STOP_SIGNAL = -1;

    if (rank == 0) {
        // --- MASTER PROCESS ---
        int next_num = 2;
        int active_workers = size - 1;

        if (size < 2) {
            std::cerr << "This program requires at least 2 processes." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        while (active_workers > 0) {
            int result;
            MPI_Status status;

            // Receive a result (or a "ready" signal) from any worker
            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            // If the worker sent a positive number, it's a prime we found
            if (result > 0) {
                std::cout << "Prime found: " << result << std::endl;
            }

            if (next_num <= MAX_VAL) {
                // Send the next number to the worker that just finished
                MPI_Send(&next_num, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                next_num++;
            } else {
                // No more work; tell this specific worker to stop
                MPI_Send(&STOP_SIGNAL, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                active_workers--;
            }
        }
    } else {
        // --- WORKER PROCESS ---
        int current_num = 0; // Initial "ready" signal to the master

        while (true) {
            // Send previous result (or initial signal) and ask for work
            MPI_Send(&current_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Receive new number to check
            MPI_Recv(&current_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (current_num == STOP_SIGNAL) break;

            // Process the work
            if (isPrime(current_num)) {
                // Keep the number as is (positive) to indicate it's prime
            } else {
                // Flip to negative to indicate it's not prime (but still send back)
                current_num = -current_num;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
