package problem1;

import java.util.*;

public class pc_static_block {
    private static int NUM_END = 200000;
    private static int NUM_THREAD = 4;

    public static void main(String[] args) throws InterruptedException {
        if (args.length == 2) {
            NUM_THREAD = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }

        long startTime = System.currentTimeMillis();
        int num = NUM_END / NUM_THREAD, counter = 0;
        List<MultiThread> threads = new ArrayList<>();

        for (int i = 0; i < NUM_THREAD; i++) {
            MultiThread thread = new MultiThread(i * num, (i + 1) * num);
            threads.add(thread);
            thread.start();
        }

        for (MultiThread thread : threads) {
            thread.join();
            counter += thread.getCounter();
        }

        long endTime = System.currentTimeMillis();
        long timeDiff = endTime - startTime;

        for (int i = 0; i < threads.size(); i++) {
            MultiThread thread = threads.get(i);
            System.out.println("Thread " + i + " Execution Time: " + thread.getThread_ExecutionTime() + "ms #Prime Counter: " + thread.getCounter());
        }
        System.out.println("Program execution time: " + timeDiff + "ms");
    }

    private static boolean isPrime(int x) {
        if (x <= 1) return false;

        for (int i = 2; i <= Math.sqrt(x); i++) {
            if (x % i == 0) return false;
        }
        return true;
    }

    private static class MultiThread extends Thread {
        private final int start, end;
        private int counter = 0;
        private long thread_executionTime;

        public MultiThread(int start, int end) {
            this.start = start;
            this.end = end;
        }

        public int getCounter() { return counter; }
        public long getThread_ExecutionTime() { return thread_executionTime; }

        @Override
        public void run() {
            long startTime = System.currentTimeMillis();
            for (int i = start; i < end; i++) {
                if (isPrime(i)) counter++;
            }
            long endTime = System.currentTimeMillis();
            thread_executionTime = endTime - startTime;
        }
    }
}
