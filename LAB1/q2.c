#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static double *alloc_mat(int N) {
    double *A = (double *)malloc((size_t)N * (size_t)N * sizeof(double));
    if (!A) { perror("malloc"); exit(1); }
    return A;
}

static void fill_random(double *A, int N) {
    for (int i = 0; i < N*N; i++) A[i] = (double)rand() / (double)RAND_MAX;
}

static void transpose(const double *Y, double *YT, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            YT[(size_t)j * N + i] = Y[(size_t)i * N + j];
}

static void matmul_seq(const double *X, const double *Y, double *Z, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += X[(size_t)i * N + k] * Y[(size_t)k * N + j];
            }
            Z[(size_t)i * N + j] = sum;
        }
    }
}

static void matmul_par(const double *X, const double *Y, double *Z, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += X[(size_t)i * N + k] * Y[(size_t)k * N + j];
            }
            Z[(size_t)i * N + j] = sum;
        }
    }
}

/* Optimised: use YT (transpose of Y) for better cache locality
   Still computes Z = X * Y using: Z[i][j] = sum_k X[i][k] * YT[j][k] */
static void matmul_opt(const double *X, const double *YT, double *Z, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            const double *xrow = &X[(size_t)i * N];
            const double *ytrow = &YT[(size_t)j * N];
            for (int k = 0; k < N; k++) {
                sum += xrow[k] * ytrow[k];
            }
            Z[(size_t)i * N + j] = sum;
        }
    }
}

int main(void)
{
    int Ns[] = {512, 1024, 2048};
    int nN = (int)(sizeof(Ns) / sizeof(Ns[0]));

    int total_threads = omp_get_max_threads();

    printf("Q2. Matrix Computation(./q2)\n\n");

    printf("+------+---------------------------+--------------------------+---------------------------+------------+\n");
    printf("|  N   |   Sequential              |   Parallel               |   Optimised (Transposed)  | Efficiency |\n");
    printf("|      | Time(sec) | Speedup       | Time(sec) | Speedup      | Time(sec) | Speedup       |           |\n");
    printf("+------+-----------+---------------+-----------+--------------+-----------+---------------+-----------+\n");

    for (int idx = 0; idx < nN; idx++) {
        int N = Ns[idx];

        double *X  = alloc_mat(N);
        double *Y  = alloc_mat(N);
        double *Z  = alloc_mat(N);
        double *YT = alloc_mat(N);

        srand(0);
        fill_random(X, N);
        fill_random(Y, N);

        /* Transpose outside timing (matches typical lab table style) */
        transpose(Y, YT, N);

        /* Sequential */
        double t0 = omp_get_wtime();
        matmul_seq(X, Y, Z, N);
        double t_seq = omp_get_wtime() - t0;

        /* Parallel */
        t0 = omp_get_wtime();
        matmul_par(X, Y, Z, N);
        double t_par = omp_get_wtime() - t0;

        /* Optimised */
        t0 = omp_get_wtime();
        matmul_opt(X, YT, Z, N);
        double t_opt = omp_get_wtime() - t0;

        double sp_par = t_seq / t_par;
        double sp_opt = t_seq / t_opt;
        double eff = (sp_opt * 100.0) / (double)total_threads;

        printf("| %4d | %9.2f | %-13s | %9.2f | %10.3fx | %9.2f | %11.3fx | %8.0f%% |\n",
               N,
               t_seq, "1x",
               t_par, sp_par,
               t_opt, sp_opt,
               eff);

        free(X); free(Y); free(Z); free(YT);
    }

    printf("+------+-----------+---------------+-----------+--------------+-----------+---------------+-----------+\n");
    printf("Total Threads = %d\n\n", total_threads);

    /* Thread scaling table for N = 1024 using optimised method */
    {
        int N = 1024;
        int threads_list[] = {1, 2, 4, 8};
        int nt = (int)(sizeof(threads_list) / sizeof(threads_list[0]));

        double *X  = alloc_mat(N);
        double *Y  = alloc_mat(N);
        double *Z  = alloc_mat(N);
        double *YT = alloc_mat(N);

        srand(0);
        fill_random(X, N);
        fill_random(Y, N);
        transpose(Y, YT, N);

        printf("+---------+----------+----------+\n");
        printf("| Threads | Time(sec)| Speedup  |\n");
        printf("+---------+----------+----------+\n");

        double base = 0.0;

        for (int i = 0; i < nt; i++) {
            int T = threads_list[i];
            omp_set_num_threads(T);

            double t0 = omp_get_wtime();
            matmul_opt(X, YT, Z, N);
            double t = omp_get_wtime() - t0;

            if (T == 1) base = t;
            double sp = base / t;

            if (T == 1)
                printf("| %7d | %8.2f | %-8s |\n", T, t, "1x");
            else
                printf("| %7d | %8.2f | %7.3fx |\n", T, t, sp);
        }

        printf("+---------+----------+----------+\n");
        printf("N = %d\n", N);

        free(X); free(Y); free(Z); free(YT);
    }

    return 0;
}
