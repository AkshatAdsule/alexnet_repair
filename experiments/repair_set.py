#!/usr/bin/env python3
import sys, os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment_runner

if __name__ == "__main__":
    # Remove the subcommand token if present
    if len(sys.argv) > 1 and sys.argv[1] == "repair_set":
        sys.argv.pop(1)
    experiment_runner.main()
