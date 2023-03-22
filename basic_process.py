import threading
import time

def my_thread_func():
    time.sleep(12)
    print("Hello from thread")

def my_thread2_func():
    print("Thread2")

# Create a thread object
my_thread = threading.Thread(target=my_thread_func)
my_thread2 = threading.Thread(target=my_thread2_func)

# Start the thread
my_thread.start()
my_thread2.start()
# Wait for the thread to finish
my_thread.join()
my_thread2.join()
