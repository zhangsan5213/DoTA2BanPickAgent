"""Profile training with cProfile - saves stats periodically and on SIGINT."""
import cProfile
import pstats
import sys
import os
import signal
import time
import threading
import atexit

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

STAT_FILE = 'train_profile.stats'
DUMP_INTERVAL = 60  # Save every 60 seconds

profiler = cProfile.Profile()


def dump_stats():
    """Dump current profiling stats to file."""
    try:
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumtime')
        stats.dump_stats(STAT_FILE)
        print(f"\n[Profile] Saved to {STAT_FILE}")
    except Exception as e:
        print(f"\n[Profile] Error saving: {e}")


def print_top(n=15):
    """Print top N functions by cumulative time."""
    print(f"\n{'='*70}")
    print(f"PROFILE SNAPSHOT (top {n} by cumtime)")
    print(f"{'='*70}")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(n)
    print(f"{'='*70}\n")


def periodic_dump():
    """Background thread that dumps stats periodically."""
    while True:
        time.sleep(DUMP_INTERVAL)
        dump_stats()
        print_top(10)


def signal_handler(signum, frame):
    print("\n[!] Signal received, saving profile...")
    dump_stats()
    print_top(20)
    sys.exit(0)


# Register handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(dump_stats)

# Start background dumper
dumper_thread = threading.Thread(target=periodic_dump, daemon=True)
dumper_thread.start()

# Import and run training
from train_bp_agent import main

profiler.enable()
try:
    main()
except KeyboardInterrupt:
    print("\n[!] Interrupted by user")
finally:
    profiler.disable()
    dump_stats()
    print_top(20)
