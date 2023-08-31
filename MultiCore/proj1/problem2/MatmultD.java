package problem2;

import java.util.*;
import java.io.*;

public class MatmultD implements Runnable {
    private static Scanner sc = new Scanner(System.in);
    private static int[][] A, B, C;
    private int start, end;

    public MatmultD(int start, int end) {
        this.start = start;
        this.end = end;
    }

    public static void main(String[] args) {
        int thread_id = 0;
        String filename = "";
        if (args.length == 2) {
            thread_id = Integer.parseInt(args[0]);
            filename = args[1];
        } else {
            thread_id = 1;
            System.out.println("Usage: java MatmultD <num_threads> <filename>");
            System.exit(1);
        }

        A = readMatrix(filename);
        B = readMatrix(filename);

        int a = A.length, b = B[0].length;
        C = new int[a][b];
        int size = a / thread_id, start = 0, end = 0;
        Thread[] MultiThread = new Thread[thread_id];
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < thread_id; i++) {
            start = end;
            end = start + size;
            if (i == thread_id - 1) {
                end = a;
            }
            MultiThread[i] = new Thread(new MatmultD(start, end));
            MultiThread[i].start();
        }
        long endTime = System.currentTimeMillis();

        for (int i = 0; i < thread_id; i++) {
            try {
                MultiThread[i].join();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        printMatrix(C);

        if (thread_id > 1) {
            A = readMatrix(filename);
            B = readMatrix(filename);

            a = A.length;
            b = B[0].length;
            C = new int[a][b];
            size = a;
            start = 0;
            end = a;
            Thread[] AllThreads = new Thread[thread_id];
            startTime = System.currentTimeMillis();

            for (int i = 0; i < thread_id; i++) {
                AllThreads[i] = new Thread(new MatmultD(start, end));
                AllThreads[i].start();
            }

            for (int i = 0; i < thread_id; i++) {
                try {
                    AllThreads[i].join();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            endTime = System.currentTimeMillis();
            System.out.printf("Execution time when using all threads: %d ms\n", endTime - startTime);
        }
    }

    public static int[][] readMatrix(String filename) {
        try {
            Scanner fileScanner = new Scanner(new File(filename));
            int rows = fileScanner.nextInt();
            int cols = fileScanner.nextInt();
            int[][] arr = new int[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    arr[i][j] = fileScanner.nextInt();
                }
            }
            fileScanner.close();
            return arr;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    public static void printMatrix(int[][] mt) {
        int rows = mt.length, cols = mt[0].length, sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += mt[i][j];
            }
        }
        System.out.println();
        System.out.println("Sum of all elements in the matrix = " + sum + "\n");
    }
    @Override
    public void run() {
        int a = A[0].length;
        long startTime = System.currentTimeMillis();
        for (int i = start; i < end; i++) {
            for (int j = 0; j < a; j++) {
                for (int k = 0; k < a; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("%s Execution Time: %d ms\n", Thread.currentThread().getName(), endTime - startTime);
    }
}