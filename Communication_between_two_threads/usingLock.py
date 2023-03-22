import threading
import time

# Create a lock object
lock = threading.Lock()

# A shared resource
shared_var = 0

def thread1_func():
    global shared_var
    # Acquire the lock
    lock.acquire()
    # Modify the shared variable
    shared_var += 1
    # Release the lock
    lock.release()

def thread2_func():
    global shared_var
    # Acquire the lock
    lock.acquire()
    # Modify the shared variable
    shared_var -= 1
    # Release the lock
    lock.release()

# Create two thread objects
thread1 = threading.Thread(target=thread1_func)
thread2 = threading.Thread(target=thread2_func)

# Start the threads
thread1.start()
thread2.start()

# Wait for the threads to finish
thread1.join()
thread2.join()

# Check the value of the shared variable
print(shared_var)
