#!/usr/bin/env python3
#
# Script to convert VASP NEB calculation to ASE-extxyz trajectory
# by Patrick Melix
#
# You can import the module and then call .main() or use it as a script
from ase import io
import os
import glob

def run(outFile='movie.xyz', workdir='.', wrap=False, use=None):
    """
        use: None -> Auto use CONTCAR if available, otherwise POSCAR
             CONTCAR or POSCAR
    """

    #if output exists mv to .bak
    if os.path.isfile(outFile):
        print('ATTENTION: {:} exists, moving to *.bak'.format(outFile))
        os.rename(outFile, outFile+'.bak')

    if use:
        filename = use
    else:
        if os.path.isfile(os.path.join(workdir,'01','CONTCAR')):
            filename = 'CONTCAR'
        elif os.path.isfile(os.path.join(workdir,'01','POSCAR')):
            filename = 'POSCAR'
        else:
            raise RuntimeError("Could neither find CONTCAR nor POSCAR in {:}".format(os.path.join(workdir,'01')))
        print("Using {:} files.".format(filename))
    mol = []
    dirs = glob.glob(os.path.join(workdir,'[0-9][0-9]'))
    dirs.sort()
    print("Found {:} NEB subdirs.".format(len(dirs)))
    print("Loading images ", end='')
    for i,image in enumerate(dirs):
        if (i == 0) or (i == len(dirs)-1):
            imagePath = os.path.join(image, 'POSCAR')
        else:
            imagePath = os.path.join(image,filename)
        if not os.path.isfile(imagePath):
            raise RuntimeError('File {:} does not exist'.format(str(imagePath)))

        print(" {:}".format(os.path.split(image)[-1]), end='')
        mol.append(io.read(imagePath, format='vasp'))

    print("")
    for frame in mol:
        if wrap:
            frame.wrap(center=(0.0,0.0,0.0))
        frame.write(outFile,append=True)
    return



def main():
    """CLI entry point: parse arguments and call run()."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert VASP NEB images to an ASE ext-xyz trajectory (like nebmovie.pl).",
        epilog="Example: neb2movie --output movie.xyz --workdir . CONTCAR")
    parser.add_argument("-o", "--output", type=str, default="movie.xyz",
                        help="Output xyz file (default: movie.xyz)")
    parser.add_argument("-i", "--workdir", type=str, default=".",
                        help="NEB working directory containing 00/, 01/, ... subdirs (default: .)")
    parser.add_argument("-w", "--wrap", help="Wrap atoms with origin as center",
                        action="store_true", default=False)
    parser.add_argument("use", type=str, nargs="?",
                        help="Force reading CONTCAR or POSCAR (default: auto-prefer CONTCAR)")
    args = parser.parse_args()
    if args.use == "CONTCAR":
        use = "CONTCAR"
    elif args.use == "POSCAR":
        use = "POSCAR"
    else:
        use = None
    run(args.output, args.workdir, args.wrap, use)


if __name__ == "__main__":
    main()


