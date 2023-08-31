package problem1;

import java.util.*;

public class pc_static_cyclic {
    private static int NUM_END = 200000;
    private static int NUM_THREAD = 4;
    private static int TASK_SIZE = 10;
    private static int TASKS_PER_THREAD = NUM_END / (NUM_THREAD * TASK_SIZE);
    private static List<List<Integer>> taskList = new ArrayList<>();

    public static void main(String[] args) throws InterruptedException {
        if (args.length == 2) {
            NUM_THREAD = Integer.parseInt(args[0]);
            NUM_END = Integer.parseInt(args[1]);
            TASKS_PER_THREAD = NUM_END / (NUM_THREAD * TASK_SIZE);
        }

        long startTime = System.currentTimeMillis();
        int counter = 0;
        createTasks();
        List<MultiThread> threads = new ArrayList<>();

        for (int i = 0; i < NUM_THREAD; i++) {
            MultiThread thread = new MultiThread(i, taskList.get(i));
            threads.add(thread);
            thread.start();
        }

        long endTime = 0;

        for (MultiThread thread : threads) {
            thread.join();
            counter += thread.getCounter();
            endTime = Math.max(endTime, thread.getEndTime());
        }

        long timeDiff = endTime - startTime;

        for (int i = 0; i < threads.size(); i++) {
            MultiThread thread = threads.get(i);
            System.out.println("Thread " + i + " Execution Time: " + thread.getExecutionTime() + "ms #Prime Counter: " + thread.getCounter());
        }
        System.out.println("Program Execution Time: " + timeDiff + "ms");
    }

    private static boolean isPrime(int x) {
        if (x <= 1) return false;

        for (int i = 2; i <= Math.sqrt(x); i++) {
            if (x % i == 0) return false;
        }
        return true;
    }

    private static void createTasks() {
        for (int i = 0; i < NUM_THREAD; i++) {
            List<Integer> task = new ArrayList<>();
            for (int j = 0; j < TASKS_PER_THREAD; j++) {
                int start = i * TASK_SIZE * TASKS_PER_THREAD + j * TASK_SIZE + 1;
                int end = start + TASK_SIZE - 1;
                if ( (i == NUM_THREAD - 1) && (j == TASKS_PER_THREAD - 1)) end = NUM_END;
                task.add(start);
                task.add(end);
            }
            taskList.add(task);
        }
    }

    private static class MultiThread extends Thread {
        private final int num;
        private final List<Integer> task;
        private int counter = 0;
        private long thread_executionTime, endTime;

        public MultiThread(int num, List<Integer> task) {
            this.num = num;
            this.task = task;
        }

        public int getCounter() { return counter; }
        public long getExecutionTime() { return thread_executionTime; }
        public long getEndTime() { return endTime; }

        @Override
        public void run() {
            long startTime = System.currentTimeMillis();
            for (int i = 0; i < task.size(); i += 2) {
                int start = task.get(i);
                int end = task.get(i + 1);
                for (int j = start; j <= end; j++) {
                    if (isPrime(j)) counter++;
                }
            }
            long currentTime = System.currentTimeMillis();
            thread_executionTime = currentTime - startTime;
            endTime = currentTime;
        }
    }
}
