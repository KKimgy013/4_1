import java.util.concurrent.atomic.AtomicInteger;

public class ex3 {
    private AtomicInteger atom = new AtomicInteger();

    public void set(int value) {
        atom.set(value);
    }

    public int get() {
        return atom.get();
    }

    public int addAndGet(int value) {
        return atom.addAndGet(value);
    }

    public int getAndAdd(int value) {
        return atom.getAndAdd(value);
    }

    public static void main(String[] args) {
        ex3 ex = new ex3();

        ex.set(19);
        System.out.println("Initial value: " + ex.get());

        int r1 = ex.addAndGet(12);
        System.out.println("addAndGet(45) method: " + r1);

        int r2 = ex.getAndAdd(34);
        System.out.println("getAndAdd(4) method: " + r2);
        System.out.println("Current value: " + ex.get());
    }
}