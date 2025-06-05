# main.py

import call
import threading
import time

# This function receives the two values
def print_result(name, score):
    print(f"Received: {name} - Score: {score}")

def start():
    threading.Thread(target=lambda: call.do_work(print_result), daemon=True).start()

def stop():
    call.is_running = False

# Demo run

start()
time.sleep(5)  # Let it run for 5 seconds
stop()
time.sleep(1)  # Wait for final message
