*Project1 
Problem 1 - JAVA program (pc_serial.java) computes the number of ‘prime numbers’ between 1 and 200000 using a single thread. 
Implement multithreaded version of pc_serial.java using static load balancing (using block decomposition), static load balancing (using cyclic decomposition), and dynamic load balancing. Submit the multithreaded JAVA codes ("pc_static_block.java", "pc_static_cyclic.java" and "pc_dynamic.java").  
- Print the (1) execution time of each thread and (2) program execution time and (3) the number of ‘prime numbers’. 

Problem 2 - a JAVA source code for matrix multiplication (the source code MatmultD.java is available on our class webpage)
Implement parallel matrix multiplication that uses multi-threads. You should use a static load balancing approach.
- Print as output (1) the execution time of each thread, (2) execution time when using all threads, and (3) sum of all elements in the resulting matrix. Use the matrix mat500.txt (available on our class webpage) as file input (standard input) for the performance evaluation. mat500.txt contains two matrices that will be used for multiplication.


*Project2
Problem 1 - JAVA code generating results that are equivalent (i.e. similar) to the results of the original JAVA code ParkingGarageOperation.java using ArrayBlockingQueue and BlockingQueue in java.util.concurrent package instead of using wait()/notify(). This means you should not use wait()/notif

Problem 2 - JAVA code generating the results that are equivalent to the results of the original JAVA code ParkingGarageOperation.java by using Semaphore class in java.util.concurrent package. This is similar to [problem 1], but the difference is that you need to do modification using Semaphore (i.e. counting semaphore class) instead of using ArrayBlockingQueue.

Problem 3
(i)-a. Explain the interface/class BlockingQueue and ArrayBlockingQueue.
(i)-b. Create and include (in your document) your own example of multithreaded JAVA code (ex1.java) that is simple and executable. Your example code should use put() and take() methods. Also, include example execution results (i.e. output) in your document. 
(ii)-a. Do the things similar to (i)-a for the class ReadWriteLock. 
(ii)-b. Do the things similar to (i)-b for lock(), unlock(), readLock() and writeLock() of ReadWriteLock. (ex2.java) 
(iii)-a. Do the things similar to (i)-a for the class AtomicInteger. 
(iii)-b. Do the things similar to (i)-b for get(), set(), getAndAdd(), and addAndGet() methods of AtomicInteger. (ex3.java) 
(iv)-a. Do the things similar to (i)-a for the class CyclicBarrier. 
(iv)-b. Do the things similar to (i)-b for await() methods of CyclicBarrier. (ex4.java)


*Project3
Problem 1 - Write ‘C with OpenMP’ code that computes the number of prime numbers between 1 and 200000
- Scheduling type number (1 = “static with default chunk size”, 2 = “dynamic with default chunk size”, 3 = “static with chunk size 10”, 4 = “dynamic with chunk size 10”)
- Number of threads (1, 2, 4, 6, 8, 10, 12, 14, 16) as program input argument.

Problem 2 - Parallelize prob2.c using OpenMP. 
- Program should take three command line arguments: 
- Scheduling type number (1=static, 2=dynamic, 3=guided), chunk size, and number of threads as program input argument. 
- Print the execution time and the result of PI calculation. 


*Project4
Problem 1 - Implementing two versions of Ray-Tracing (openmp_ray.c, and cuda_ray.cu) that utilizes OpenMP and CUDA library, respectively.

Problem 2 -  Use thrust that is an open-source library.
- Approximate computation of the integral by computing a sum of rectangles where each rectangle has width delta-x and height F(xi) at the middle of interval and F(x)= 4.0 / (1+x^2). 
- Choose N=1,000,000,000 that is a sufficiently large number for experimentation.
