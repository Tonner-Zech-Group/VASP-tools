#!/usr/bin/env python3
import subprocess
import os


#simple wrapper for calc-deformation-density.sh script

#
def main():
    script_path = os.path.join(os.path.dirname(__file__), 'bash_scripts/calc-deformation-density.sh')
    subprocess.run([script_path], check=True)

if __name__ == "__main__":
    main()
