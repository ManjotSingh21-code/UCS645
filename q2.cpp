#include <iostream>
#include <cstring>
#include <omp.h>

inline int max3(int a, int b, int c) {
    return std::max(a, std::max(b, c));
}

int main(int argc, char* argv[]) {

    int threads = 0;

    if (argc > 1) {
        threads = std::atoi(argv[1]);
        omp_set_num_threads(threads);
    }

    if (threads == 0) {
        threads = omp_get_max_threads();
    }

    const char A[] = "ACACACTA";
    const char B[] = "AGCACACA";

    int m = std::strlen(A);
    int n = std::strlen(B);

    int H[m + 1][n + 1];

    int match = 2;
    int mismatch = -1;
    int gap = -1;

    for (int i = 0; i <= m; i++)
        for (int j = 0; j <= n; j++)
            H[i][j] = 0;

    double start = omp_get_wtime();

    for (int d = 1; d <= m + n - 1; d++) {

        #pragma omp parallel for
        for (int i = 1; i <= m; i++) {

            int j = d - i;

            if (j >= 1 && j <= n) {

                int score = (A[i - 1] == B[j - 1]) ? match : mismatch;

                int diag = H[i - 1][j - 1] + score;
                int up   = H[i - 1][j] + gap;
                int left = H[i][j - 1] + gap;

                H[i][j] = std::max(0, max3(diag, up, left));
            }
        }
    }

    double end = omp_get_wtime();

    std::cout << "Smith-Waterman completed\n";
    std::cout << "Threads Used: " << threads << "\n";
    std::cout << "Execution Time: " << (end - start) << " seconds\n";

    return 0;
}
