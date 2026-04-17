#!/usr/bin/env python3
"""Wrapper to run the getPOTCAR.sh bash script as an installed console command."""
import subprocess
import sys
import os


def main():
    script_path = os.path.join(os.path.dirname(__file__), 'bash_scripts/getPOTCAR.sh')
    result = subprocess.run([script_path] + sys.argv[1:], check=False)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
