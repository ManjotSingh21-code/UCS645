#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <omp.h>

#include "correlate.h"

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <ny_rows> <nx_cols> [threads]\n";
        return 0;
    }

    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);

    int threads = omp_get_max_threads();
    if (argc >= 4) threads = std::atoi(argv[3]);

    omp_set_num_threads(threads);

    std::cout << "Matrix: ny=" << ny << ", nx=" << nx << "\n";
    std::cout << "Threads: " << threads << "\n";

    std::vector<float> data(ny * nx);
    std::vector<float> result(ny * ny, 0.0f);

    // Fill data with random numbers
    for (int i = 0; i < ny * nx; i++)
        data[i] = (float)(rand() % 100) / 10.0f;

    auto start = std::chrono::high_resolution_clock::now();

    correlate(ny, nx, data.data(), result.data());

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Time taken: " << ms << " ms\n";

    // print a small sample
    std::cout << "Sample output (0,0): " << result[0] << "\n";
    std::cout << "Sample output (1,0): " << result[1] << "\n";
    std::cout << "Sample output (1,1): " << result[1 + 1*ny] << "\n";

    return 0;
}
