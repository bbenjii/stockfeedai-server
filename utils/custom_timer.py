import time


class CustomTimer:
    def __init__(self, name="Timer"):
        self.name = name
        self._start_time = None
        self._elapsed_times = []

    def start(self):
        """Starts the timer."""
        if self._start_time is not None:
            print(f"Warning: {self.name} is already running. Stopping and restarting.")
            self.stop()
        self._start_time = time.perf_counter()
        print(f"----------{self.name} started.----------")

    def stop(self):
        """Stops the timer and records the elapsed time."""
        if self._start_time is None:
            raise RuntimeError(f"{self.name} has not been started.")

        end_time = time.perf_counter()
        elapsed = end_time - self._start_time
        self._elapsed_times.append(elapsed)
        self._start_time = None  # Reset for next start
        print(f"----------{self.name} stopped. Elapsed time: {elapsed:.4f} seconds.----------")
        return elapsed

    def reset(self):
        """Resets the timer, clearing all recorded elapsed times."""
        self._start_time = None
        self._elapsed_times = []
        print(f"{self.name} reset.")

    @property
    def total_elapsed(self):
        """Returns the sum of all recorded elapsed times."""
        return sum(self._elapsed_times)

    @property
    def average_elapsed(self):
        """Returns the average of all recorded elapsed times."""
        if not self._elapsed_times:
            return 0
        return self.total_elapsed / len(self._elapsed_times)

    def print_stats(self):
        """Prints statistics about the timer's performance."""
        print(f"\n--- {self.name} Statistics ---")
        print(f"Total elapsed time: {self.total_elapsed:.4f} seconds")
        print(f"Number of runs: {len(self._elapsed_times)}")
        print(f"Average elapsed time: {self.average_elapsed:.4f} seconds")
        print("------------------------------")


# Example Usage
# if __name__ == "__main__":
#     my_timer = CustomTimer("My Code Timer")
# 
#     my_timer.start()
#     time.sleep(0.5)  # Simulate some work
#     my_timer.stop()
# 
#     my_timer.start()
#     time.sleep(0.02)
#     my_timer.stop()
# 
#     my_timer.print_stats()
# 
#     # Another timer for a different task
#     another_timer = CustomTimer("Task B Timer")
#     another_timer.start()
#     time.sleep(0.1)
#     another_timer.stop()
#     another_timer.print_stats()