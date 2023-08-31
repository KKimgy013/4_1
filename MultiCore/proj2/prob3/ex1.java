import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;

class Sender implements Runnable {
    private BlockingQueue<String> queue;
    private int num;
    public Sender(BlockingQueue<String> queue, int num) {
        this.queue = queue;
        this.num = num;
    }

    public void run() {
        while (true) {
            try {
                String message = "Hi I'm #" + num + " sender. Who R U?";
                queue.put(message);
                System.out.println("Sender " + num + " write message: " + message);
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

class Receiver implements Runnable {
    private BlockingQueue<String> queue;
    private int num;
    public Receiver(BlockingQueue<String> queue, int num) {
        this.queue = queue;
        this.num = num;
    }

    public void run() {
        while (true) {
            try {
                String message = queue.take();
                System.out.println("Receiver " + num + " got message: " + message + "       Receiver says: Hi I'm #" + num + " receiver.");
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

public class ex1 {
    public static void main(String[] args) {
        BlockingQueue<String> queue = new ArrayBlockingQueue<String>(10);

        for (int i = 1; i <= 3; i++) {
            Sender s = new Sender(queue, i);
            Thread t = new Thread(s);
            t.start();
        }

        for (int i = 1; i <= 5; i++) {
            Receiver r = new Receiver(queue, i);
            Thread t = new Thread(r);
            t.start();
        }
    }
}
