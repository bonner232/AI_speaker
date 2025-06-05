# worker.py

import time

is_running = False

def do_work(callback):
    global is_running
    is_running = True

    while is_running:
        # Simulate generating two values
        name = "Alice" if time.time() % 2 < 1 else "Bob"
        score = round(50 + (time.time() % 10), 2)

        # Call the function passed in
        callback(name, score)

        time.sleep(1)

    callback("Stopped", 0)
