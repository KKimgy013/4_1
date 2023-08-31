#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int stn, char *arr[]) {
    if (stn != 4) {
        printf("Usage: %s scheduling_type_number chunk_size num_of_threads\n", arr[0]);
        return 1;
    }

    int num_steps = 10000000;
    int scheduling_type = atoi(arr[1]);
    int chunk_size = atoi(arr[2]);
    int num_threads = atoi(arr[3]);

    double step = 1.0 / (double)num_steps;
    double pi = 0.0;

    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();
    double x, sum = 0.0;

    #pragma omp parallel reduction(+:sum) private(x)
    {
        if(scheduling_type == 1){
            #pragma omp for schedule(static, chunk_size)
            for(int i=0; i<num_steps; i++) {
                x = (i + 0.5) * step;
                sum += 4.0 / (1.0 + x * x);
            }
        }
        else if(scheduling_type == 2) {
            #pragma omp for schedule(dynamic, chunk_size)
            for(int i=0; i<num_steps; i++) {
                x = (i + 0.5) * step;
                sum += 4.0 / (1.0 + x * x);
            }
        }
        else if (scheduling_type == 3) {
            #pragma omp for schedule(guided, chunk_size)
            for(int i=0; i<num_steps; i++) {
                x = (i + 0.5) * step;
                sum += 4.0 / (1.0 + x * x);
            }
        }
        pi += sum * step;
    }

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    printf("Result of PI calculation: %.24lf\n", pi);
    printf("Execution time: %lf ms\n", execution_time);

    return 0;
}
