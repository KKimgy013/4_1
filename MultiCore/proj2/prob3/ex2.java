import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ex2 {
    private int value = 0;
    private ReadWriteLock rw = new ReentrantReadWriteLock();

    public void jump() {
        rw.writeLock().lock();
        try {
            value++;
        } finally {
            rw.writeLock().unlock();
        }
    }

    public int getVariable() {
        rw.readLock().lock();
        try {
            return value;
        } finally {
            rw.readLock().unlock();
        }
    }

    public static void main(String[] args) {
        ex2 ex = new ex2();

        for (int i = 1; i <= 5; i++) {
            Thread t = new Thread(() -> {
                for (int j = 0; j < 12345; j++) {
                    ex.jump();
                }
            });
            t.start();
        }

        Thread t = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Shared value: " + ex.getVariable());
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        t.start();
    }
}
