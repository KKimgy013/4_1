import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class ex4 {
    private static final int THREAD_NUM = 5;
    private static final CyclicBarrier bar = new CyclicBarrier(THREAD_NUM, new Runnable() {
        @Override
        public void run() {
            System.out.println("All threads Finished! Thank U");
        }
    });

    public static void main(String[] args) {
        for (int i = 1; i <= THREAD_NUM; i++) {
            Thread t = new Thread(new DoThread(i));
            t.start();
        }
    }

    private static class DoThread implements Runnable {
        private final int id;

        public DoThread(int id) {
            this.id = id;
        }

        public void run() {
            System.out.println("Thread " + id + " starts!");
            try {
                Thread.sleep(1000 + (long) (Math.random() * 2000));
                System.out.println("Thread " + id + " finishes!");
                bar.await();
            } catch (InterruptedException | BrokenBarrierException e) {
                e.printStackTrace();
            }
        }
    }
}
