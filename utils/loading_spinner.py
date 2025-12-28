import time
import sys
from utils.logger import logger

def simple_spinner():
    symbols = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']
    i = 0
    dot_count = 1
    max_dots = 4  # From "." to "...."

    while True:
        # Update spinner index
        i = (i + 1) % len(symbols)

        # Build the animated dots string
        dots = '.' * dot_count
        dot_count = (dot_count % max_dots) + 1  # Cycle 1 → max_dots

        # Print the spinner + animated dots
        print(f'\r\033[K loading{dots}', flush=True, end='')

        time.sleep(0.2)
simple_spinner()