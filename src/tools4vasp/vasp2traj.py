#!/usr/bin/env python3
#
# Script to convert VASP output or XDATCAR to ASE-extxyz trajectory
# by Patrick Melix
# 2018/03/13
#
# You can import the module and then call .main() or use it as a script
from ase import io
import os

def run(outFile, inFiles, wrap):
    #if output exists mv to .bak
    if os.path.isfile(outFile):
        print('ATTENTION: {:} exists, moving to *.bak'.format(outFile))
        os.rename(outFile, outFile+'.bak')

    for inFile in inFiles:
        if not os.path.isfile(inFile):
            raise ValueError('File {:} does not exist'.format(str(inFile)))


        if "xdatcar" in inFile.lower():
            mol = io.read(inFile, format='vasp-xdatcar', index=slice(0,None))
        else:
            mol = io.read(inFile, format='vasp-out', index=slice(0,None))

        for frame in mol:
            if wrap:
                frame.wrap(center=(0.0,0.0,0.0))
            frame.write(outFile,append=True)
    return



def main():
    """CLI entry point: parse arguments and call run()."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert VASP geometry optimisation output (OUTCAR or XDATCAR) to an "
                    "ASE ext-xyz trajectory file.",
        epilog="Example: vasp2traj traj.xyz OUTCAR")
    parser.add_argument("-w", "--wrap", help="Wrap atoms with origin as center",
                        action="store_true", default=False)
    parser.add_argument("output", type=str, help="Output xyz trajectory file")
    parser.add_argument("input", type=str, nargs="*",
                        help="Input VASP file(s): OUTCAR or XDATCAR")
    args = parser.parse_args()
    run(args.output, args.input, args.wrap)


if __name__ == "__main__":
    main()


