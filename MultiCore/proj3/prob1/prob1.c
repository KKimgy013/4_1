#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

bool is_prime(int num){
    if(num <= 1) return false;

    for(int i=2; i*i<= num; i++) {
        if(num%i == 0) return false;
    }
    return true;
}

int main(int stn, char *arr[]) {
    if (stn != 3) {
        printf("Usage: %s scheduling_type_number num_of_threads\n", arr[0]);
        return 1;
    }

    int scheduling_type = atoi(arr[1]);
    int num_threads = atoi(arr[2]);
    int cnt = 0;
    double start_time = omp_get_wtime();

    #pragma omp parallel num_threads(num_threads)
    {
        if(scheduling_type == 1){
            #pragma omp for schedule(static) reduction(+:cnt)
            for (int i=1; i<=200000; i++) {
                if(is_prime(i)) cnt++;
            }
        }
        else if(scheduling_type == 2){
            #pragma omp for schedule(dynamic) reduction(+:cnt)
            for(int i=1; i<=200000; i++) {
                if(is_prime(i)) cnt++;
            }
        }
        else if(scheduling_type == 3){
            #pragma omp for schedule(static, 10) reduction(+:cnt)
            for(int i=1; i<=200000; i++) {
                if(is_prime(i)) cnt++;
            }
        }
        else if(scheduling_type == 4){
            #pragma omp for schedule(dynamic, 10) reduction(+:cnt)
            for(int i=1; i<=200000; i++) {
                if(is_prime(i)) cnt++;
            }
        }
    }

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    printf("Number of primes: %d\n", cnt);
    printf("Execution time: %lf ms\n", execution_time);

    return 0;
}
