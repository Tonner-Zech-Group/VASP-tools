#!/usr/bin/env python3
#
# Script to plot VASP+TST NEB calculation results
# by Patrick Melix
# 2022/01
#
# You can import the module and then call .main() or use it as a script
# Needs grep and tail.
import argparse
import os
import subprocess
import numpy as np
from matplotlib.ticker import MaxNLocator
from ase.units import create_units
from matplotlib import pyplot as plt


def plot(reactionCoord, reactionCoordImageAxis, energies, energySpline, forces, filename, lw=3, s=0, highlight=None, dispersion=None, unit="kJ/mol"):
    # unit label fix
    if "/" in unit:
        parts = unit.split("/")
        assert len(parts) == 2, "Only one division allowed for unit conversion."
        unit_label = r"$\mathrm{" + parts[0] + "}" + r"\,\mathrm{" + parts[1] + r"}^{-1}$"
    msbig = 9
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Reaction Coordinate')# [Å]
    ax.set_xticklabels([]) #no numbers on x
    plt.ylabel(f'$\Delta E$ [{unit_label}]')
    #plt.ylim([-10**exp,10**exp])
    #plt.yscale('symlog')
    #plt.gca().yaxis.grid(True)
    plt.plot(reactionCoord, energySpline, color='black', ls=':', label='Cubic Spline', lw=lw)
    plt.scatter(reactionCoordImageAxis, energies, marker='P', color='red', s=(msbig+s)**2, label='NEB Energy')
    dScale = 0.02
    maxX = max(reactionCoordImageAxis)
    delta = dScale*maxX
    yRange = max(energySpline) - min(energySpline)
    for i,x in enumerate(reactionCoordImageAxis):
        tangentX = [x-delta, x+delta]
        tangentY = [energies[i]+(delta*forces[i]),energies[i]-(delta*forces[i])] #invert sign of forces from neb output
        if i == 0:
            label = 'NEB Force'
        else:
            label = None
        # limit y range of tangent
        tangentYRange = max(tangentY) - min(tangentY)
        factor = delta * 0.1
        n = 1
        while tangentYRange > 0.1 * yRange:
            delta -= factor
            tangentX = [x-delta, x+delta]
            tangentY = [energies[i]+(delta*forces[i]),energies[i]-(delta*forces[i])] #invert sign of forces from neb output
            tangentYRange = max(tangentY) - min(tangentY)
            n += 1
            if n >= 11:
                raise ValueError("Tangent Problem")
        plt.plot(tangentX, tangentY, color='green', ls='-', lw=lw, label=label)

    if highlight is not None:
        plt.scatter(reactionCoordImageAxis[highlight], energies[highlight], marker='o', s=(msbig+s+30)**2, facecolors='none', edgecolors='orange', lw=lw+2, clip_on=False)

    if dispersion is not None:
        plt.scatter(reactionCoordImageAxis[1:], dispersion[1:], color='brown', marker='o', label='Dispersion', s=(msbig+s)**2)
    #plt.xticks(x, printDirs[:], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    #plt.show()
    plt.close()


def main(filename='NEB.png', presentation=False,
            highlight=None, plot_all=False, plot_dispersion=False,
            unit='kJ/mol', load_dispersion=None):
    unitDict = create_units('2014')
    conv = unitDict['eV'] #VASP and TST use eV
    if '/' in unit:
        tmp = unit.split('/')
        conv *= unitDict[tmp[0]]
        for u in tmp[1:]:
            conv /= unitDict[u]
    else:
        conv *= unitDict[unit]
    print("Unit conversion factor from eV to {}: {:}".format(unit, conv))

    if load_dispersion:
        import json
        with open(load_dispersion, 'r') as f:
            dispersion_loaded = np.array(json.load(f))
    else:
        dispersion_loaded = None

    spline = np.loadtxt('spline.dat')
    print("Spline loaded.")
    nebData = np.loadtxt('neb.dat')
    print("Energy and forces loaded.")
    nImages = len(nebData)
    print("{:} data points found.".format(nImages))

    reactionCoord = [ s[1] for s in spline ]
    #images = [ d[0] for d in nebData ]
    reactionCoordImageAxis = [ d[1] for d in nebData ]
    #reactionCoordImageAxis = [ reactionCoord[xAxis.index(n)] for n in images ]
    energies = [ d[2]/conv for d in nebData ]
    forces = [ d[3]/conv for d in nebData ]
    energySpline = [ s[2]/conv for s in spline ]

    if presentation:
        lw = 5
        s = 3
        plt.rcParams.update({'font.size': 22})
        plt.rcParams.update({'legend.fontsize': 22})
    else:
        lw = 3
        s = 0

    dispersion = None
    if plot_dispersion and dispersion_loaded is not None:
        dispersion = dispersion_loaded
        dispersion -= dispersion[0]
        assert len(dispersion) == nImages, "Dispersion data does not match number of images."
    elif plot_dispersion:
        print("Collecting dispersion energies from OUTCARs.")
        dispersion = []
        for i in range(nImages):
            path = "{:02d}".format(i)
            assert os.path.isdir(path), "Could not find dir {}".format(path)
            outcar = os.path.join(path,'OUTCAR')
            assert os.path.isfile(outcar), "Could not find file {}".format(outcar)
            child = subprocess.Popen(["grep 'Edisp (eV)' {:} | tail -1".format(outcar)], stdout=subprocess.PIPE, shell=True)
            dispE = float(child.communicate()[0].strip().split()[-1])
            dispersion.append(dispE)
        dispersion = np.array(dispersion)
        dispersion -= dispersion[0]
        dispersion /= conv

    plot(reactionCoord, reactionCoordImageAxis, energies, energySpline, forces, filename, lw=lw, s=s, highlight=highlight, dispersion=dispersion, unit=unit)

    if plot_all:
        #plot the main image and then one with every point highlighted
        filename = filename.split('.')
        filename[-2] += "-{:02d}"
        filename = ".".join(filename)
        for i in range(nImages):
            plot(reactionCoord, reactionCoordImageAxis, energies, energySpline, forces, filename.format(i), lw=lw, s=s, highlight=i, dispersion=dispersion, unit=unit)


if __name__ == "__main__":
    exec(open("/home/patrickm/git/Python4ChemistryTools/mpl-settings.py").read())
    parser = argparse.ArgumentParser(description='Plot VASP+TST NEB results')
    parser.add_argument('--file', help='Plot Filename', default='NEB.png')
    parser.add_argument('--presentation', help='Presentation Mode (i.e. thicker lines)', action='store_true')
    parser.add_argument('--highlight', help='Circle Point N', type=int, default=None)
    parser.add_argument('--plotall', help='Create main plot and each highlighted plot.', action='store_true')
    parser.add_argument('--plotdispersion', help='Include dispersion contributions in plot.', action='store_true')
    parser.add_argument('--unit', help='Set the unit used to plot, must be ase compatible.', default='kJ/mol')
    parser.add_argument('--load-dispersion', help='Load dispersion energies from json.', type=str, default=None)
    args = parser.parse_args()
    main(args.file, args.presentation, args.highlight, args.plotall, args.plotdispersion, args.unit, args.load_dispersion)
