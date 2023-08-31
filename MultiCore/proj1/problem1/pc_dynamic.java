package problem1;

import java.util.*;

public class pc_dynamic {
    private static int NUM_END = 200000;
    private static int NUM_THREAD = 4;
    private static final int TASK_SIZE = 10;
    private static final Object lock = new Object();

    public static void main(String[] args) throws InterruptedException {
        if (args.length == 2) {
            NUM_THREAD = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
        }

        long startTime = System.currentTimeMillis();
        int range = NUM_END / (NUM_THREAD * TASK_SIZE), counter = 0;
        List<MultiThread> threads = new ArrayList<>();

        for (int i = 0; i < NUM_THREAD; i++) {
            int start = i * range * TASK_SIZE;
            int end = (i == NUM_THREAD - 1) ? NUM_END : start + range * TASK_SIZE;
            threads.add(new MultiThread(start, end));
        }

        for (MultiThread thread : threads) {
            thread.start();
        }

        for (MultiThread thread : threads) {
            thread.join();
            synchronized (lock) {
                counter += thread.getCounter();
                System.out.println(thread.getName() + " execution time: " + thread.getExecutionTime() + "ms #prime counter: " + thread.getCounter());
            }
        }
        long endTime = System.currentTimeMillis();
        long timeDiff = endTime - startTime;
        System.out.println("Program Execution Time: " + timeDiff + "ms");
    }

    private static boolean isPrime(int x) {
        if (x <= 1) return false;

        for (int i = 2; i <= Math.sqrt(x); i++) {
            if (x % i == 0) return false;
        }
        return true;
    }

    static class MultiThread extends Thread {
        private final int start, end;
        private int counter;
        private long executionTime;

        public MultiThread(int start, int end) {
            this.start = start;
            this.end = end;
        }

        public int getCounter() { return counter; }
        public long getExecutionTime() { return executionTime; }

        @Override
        public void run() {
            long startTime = System.currentTimeMillis();
            for (int i = start; i < end; i++) {
                if (isPrime(i)) {
                    synchronized (lock) {
                        counter++;
                    }
                }
            }
            long endTime = System.currentTimeMillis();
            executionTime = endTime - startTime;
        }
    }


}
