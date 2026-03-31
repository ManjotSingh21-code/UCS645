#include "correlate.h"
#include <cmath>
#include <vector>
#include <omp.h>

// Change this to 0 if you want ONLY sequential
#define USE_OMP 1

void correlate(int ny, int nx, const float* data, float* result)
{
    // Precompute mean and norm for each row
    std::vector<double> mean(ny, 0.0);
    std::vector<double> norm(ny, 0.0);

    // ----------------------------
    // Step 1: mean + norm per row
    // ----------------------------
#if USE_OMP
#pragma omp parallel for
#endif
    for (int i = 0; i < ny; i++)
    {
        double sum = 0.0;
        for (int x = 0; x < nx; x++)
            sum += (double)data[x + i * nx];

        mean[i] = sum / (double)nx;

        double sq = 0.0;
        for (int x = 0; x < nx; x++)
        {
            double v = (double)data[x + i * nx] - mean[i];
            sq += v * v;
        }

        norm[i] = std::sqrt(sq);
    }

    // ---------------------------------------
    // Step 2: correlation for all (i,j), j<=i
    // ---------------------------------------
#if USE_OMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double dot = 0.0;

            // dot product of normalized centered rows
            for (int x = 0; x < nx; x++)
            {
                double ai = (double)data[x + i * nx] - mean[i];
                double aj = (double)data[x + j * nx] - mean[j];
                dot += ai * aj;
            }

            double denom = norm[i] * norm[j];

            // avoid divide-by-zero
            double corr = (denom == 0.0) ? 0.0 : (dot / denom);

            result[i + j * ny] = (float)corr;
        }
    }
}
