#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SPHERES 20
#define DIM 2048

struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INFINITY;
    }
};

__global__ void kernel(Sphere *spheres, unsigned char *bitmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * DIM;

    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);
    float r = 0, g = 0, b = 0;
    float maxz = -INFINITY;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = spheres[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = spheres[i].r * fscale;
            g = spheres[i].g * fscale;
            b = spheres[i].b * fscale;
            maxz = t;
        }
    }
    bitmap[offset * 4 + 0] = (int)(r * 255);
    bitmap[offset * 4 + 1] = (int)(g * 255);
    bitmap[offset * 4 + 2] = (int)(b * 255);
    bitmap[offset * 4 + 3] = 255;
}

void ppm_write(unsigned char *bitmap, int xdim, int ydim, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", xdim, ydim);
    fwrite(bitmap, 1, xdim * ydim * 4, fp);
    fclose(fp);
}

int main() {
    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = static_cast<float>(rand()) / RAND_MAX;
        temp_s[i].g = static_cast<float>(rand()) / RAND_MAX;
        temp_s[i].b = static_cast<float>(rand()) / RAND_MAX;
        temp_s[i].x = static_cast<float>(rand() % 2000 - 1000);
        temp_s[i].y = static_cast<float>(rand() % 2000 - 1000);
        temp_s[i].z = static_cast<float>(rand() % 2000 - 1000);
        temp_s[i].radius = static_cast<float>(rand() % 160 + 40);
    }

    unsigned char *bitmap = (unsigned char *)malloc(sizeof(unsigned char) * DIM * DIM * 4);
    memset(bitmap, 0, sizeof(unsigned char) * DIM * DIM * 4);

    Sphere *d_spheres;
    unsigned char *d_bitmap;

    cudaMalloc((void **)&d_spheres, sizeof(Sphere) * SPHERES);
    cudaMalloc((void **)&d_bitmap, sizeof(unsigned char) * DIM * DIM * 4);

    cudaMemcpy(d_spheres, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

    dim3 dimGrid(DIM / 16, DIM / 16);
    dim3 dimBlock(16, 16);

    clock_t start = clock();

    kernel<<<dimGrid, dimBlock>>>(d_spheres, d_bitmap);

    cudaMemcpy(bitmap, d_bitmap, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);

    clock_t end = clock();
    double elapsed_sec = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    printf("CUDA ray tracing: %.3f sec\n", elapsed_sec);
    ppm_write(bitmap, DIM, DIM, "result.ppm");
    printf("[result.ppm] was generated.\n");

    cudaFree(d_spheres);
    cudaFree(d_bitmap);
    free(bitmap);
    free(temp_s);

    return 0;
}