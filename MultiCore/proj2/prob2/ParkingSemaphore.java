import java.util.concurrent.Semaphore;
class ParkingGarage2 {
    private Semaphore s;
    public ParkingGarage2(int places) {
        if (places < 0)
            places = 0;
        s = new Semaphore(places, true);
    }

    public void enter() { // enter parking garage
        try {
            s.acquire();
        }catch (InterruptedException e){}
    }

    public void leave() { // leave parking garage
        s.release();
    }

    public int getPlaces() {
        return s.availablePermits();
    }
}

class Car2 extends Thread {
    private ParkingGarage2 parkingGarage2;
    public Car2(String name, ParkingGarage2 p) {
        super(name);
        this.parkingGarage2 = p;
        start();
    }

    private void tryingEnter() {
        System.out.println(getName()+": trying to enter");
    }

    private void justEntered() {
        System.out.println(getName()+": just entered");
    }

    private void aboutToLeave() {
        System.out.println(getName()+":                                     about to leave");
    }

    private void Left() {
        System.out.println(getName()+":                                     have been left");
    }

    public void run() {
        while (true) {
            try {
                sleep((int)(Math.random() * 10000)); // drive before parking
            } catch (InterruptedException e) {}

            tryingEnter();
            parkingGarage2.enter();
            justEntered();

            try {
                sleep((int)(Math.random() * 20000)); // stay within the parking garage
            } catch (InterruptedException e) {}

            aboutToLeave();
            parkingGarage2.leave();
            Left();
        }
    }
}

public class ParkingSemaphore {
    public static void main(String[] args){
        ParkingGarage2 parkingGarage2 = new ParkingGarage2(7);
        for (int i=1; i<= 10; i++) {
            Car2 c = new Car2("Car "+i, parkingGarage2);
        }
    }
}
