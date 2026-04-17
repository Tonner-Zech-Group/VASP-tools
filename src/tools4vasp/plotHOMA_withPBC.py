#!/usr/bin/env python3
#
# Script to calculate and plot values of the Harmonic Oscillator Model of Aromaticity (HOMA),
# including a simple workaround to plot periodic boundary conditions 
# by Jakob Schramm
# 2026/03/26
#
import argparse
from ase.io import read
from ase.neighborlist import neighbor_list
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
import itertools
plt.rcParams["font.family"] = "arial"
# import timeit

def main(Coordinates, filename, C1, C2, a, d_opt, norm, rings, pbc_cutoff, max_path_len, no_of_cyc_combs, atom_types, pbc=False, no_values=False):
    mol = read(Coordinates)
    if filename == "Coordinates.svg":
        filename=Coordinates.split(".")[0]+".svg"
    #
    carbons = Atoms(None)
    for atom in mol:
        for typ in atom_types:
            if atom.symbol == typ:
                carbons.append(atom)
    # print(carbons)
    if pbc:
        max_X_coord = max([i[0] for i in carbons.get_positions()])
        min_X_coord = min([i[0] for i in carbons.get_positions()])
        max_Y_coord = max([i[1] for i in carbons.get_positions()])
        min_Y_coord = min([i[1] for i in carbons.get_positions()])

        periodic_copy = Atoms(None)
        cell = mol.get_cell()[:]
        for x in [0, 1, -1]:
            for y in [0, 1, -1]:
                for atom in carbons:
                    new_copy = Atoms(atom.symbol, [list(atom.position+x*cell[0]+y*cell[1])])
                    if (list(atom.position+x*cell[0]+y*cell[1])[0] < max_X_coord+pbc_cutoff and
                        list(atom.position+x*cell[0]+y*cell[1])[0] > min_X_coord-pbc_cutoff and
                        list(atom.position+x*cell[0]+y*cell[1])[1] < max_Y_coord+pbc_cutoff and
                        list(atom.position+x*cell[0]+y*cell[1])[1] > min_Y_coord-pbc_cutoff):
                        periodic_copy += new_copy 

        carbons = periodic_copy
    
    nl1 = neighbor_list('i', carbons, 1.6)
    nl2 = neighbor_list('j', carbons, 1.6)

    nl_dict = {}
    for i in set(nl1):
        nl_dict[i] = []
        for pos,val in enumerate(nl1):
            if i == val:
                nl_dict[i].append(nl2[pos])
    # print(nl_dict)
    # print()

    # find all existing edges in graph
    edges = []
    for i in list(nl_dict):
        for j in nl_dict[i]:
            if sorted([i,j]) not in edges:
                edges.append(sorted([i,j]))
    # print(edges)
    # print(len(edges))
    # print()

    #print("Graph loaded:", np.round(timeit.default_timer() - start_time,2))
    
    # find all paths between two points in graph
    def find_all_paths(graph, start, end, path =[]):
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        for node in graph[start]:
            if max_path_len != 0:
                if len(path) > max_path_len: #7
                    break
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        paths.sort(key=len)
        return paths
    # print(find_all_paths(nl_dict, 0, 1))
    # print()
    
    if rings == []:
        # find all possible cycles using first step of Hortons algorithm,
        # i.e. for every point (v) and independent edge (wx) combination, 
        # all possible cycles using all possible paths from v to w and v to x are created,
        # and if it is a true cycle and not noticed yet, it is appended to a list and sorted from smallest to largest
        cycles = []
        cycles_sorted = []
        for s in list(nl_dict):
            for edge in edges:
                if s not in edge:
                    path1 = find_all_paths(nl_dict, s, edge[0])
                    path2 = find_all_paths(nl_dict, s, edge[1])
                    for p in path1:
                        for q in path2:
                            if len(p) + len(q) < max_path_len+2:
                                possible_cycle = p+q[::-1]
                                possible_cycle.pop(-1)
                                if len(set(possible_cycle)) == len(possible_cycle):
                                    cycle_sorted = sorted(possible_cycle)
                                    if cycle_sorted not in cycles_sorted:
                                        cycles.append(possible_cycle)
                                        cycles_sorted.append(cycle_sorted)
        cycles.sort(key=len)
        # print(cycles)
        # print()

        # print("All cycles found:", np.round(timeit.default_timer() - start_time,2))

        # for all cycles, a list of edges for the cycles is created
        edges_of_cycles = []
        for pos1,cycle in enumerate(cycles):
            edges_of_cycles.append([])
            for pos2,v in enumerate(cycle):
                if pos2 < len(cycle)-1:
                    edges_of_cycles[pos1].append(sorted([cycle[pos2],cycle[pos2+1]]))
                else:
                    edges_of_cycles[pos1].append(sorted([cycle[pos2],cycle[0]]))
            edges_of_cycles[pos1].sort()
        # print(edges_of_cycles)
        # print()

        # Smallest Set of Smallest Rings is found by appending smallest cycle
        # if it is not a linear combination of previosly appendet smallest cycles
        SSSR = [cycles[0]]
        SSSR_edges = [edges_of_cycles[0]]
        for pos,i in enumerate(edges_of_cycles):
            helper = False
            if no_of_cyc_combs == 0:
                no_of_cyc_combs = len(SSSR_edges) + 1
            for L in range(no_of_cyc_combs):
                for subset in itertools.combinations(SSSR_edges, L):
                    comb_list = []
                    val_list=[]
                    for value in list(subset):
                        val_list.append(SSSR_edges.index(value))
                    for c in val_list:
                        comb_list += SSSR_edges[c]
                    comb = []
                    for c in comb_list:
                        if c not in comb:
                            comb.append(c)
                        else:
                            comb.remove(c)
                    comb.sort()
                    if i == comb:
                        helper = True
            if not helper:
                SSSR_edges.append(i)
                SSSR.append(cycles[pos])
                # print(len(SSSR))
        # print(SSSR_edges)
        print("Automatically determined rings:", SSSR) #, np.round(timeit.default_timer() - start_time,2))
        #print(np.round(timeit.default_timer() - start_time,2))
    else:
        SSSR = []
        for r in rings:
            SSSR.append(list(map(int, r)))
        print("Manually specified rings:", SSSR)

    size = 12
    
    shift = carbons[int(C1)].position.copy()
    for atom in carbons:
        atom.position=atom.position - shift

    x = carbons[int(C2)].position[0]
    y = carbons[int(C2)].position[1]
    if np.isclose(x, 0.0) and np.isclose(y, 0.0):
        print("ERROR: Cannot determine alignment angle because the selected alignment atom is at the origin!")
        exit()

    angle_y = np.degrees(np.arctan2(x, y))
    angle_x = np.degrees(np.arctan2(-y, x))
    if a == "y":
        rotated_y = x*np.sin(angle_y*np.pi/180)+y*np.cos(angle_y*np.pi/180)
        if rotated_y < 0:
            angle = angle_y
        elif rotated_y > 0:
            angle = 180+angle_y
        else:
            angle = angle_y
    elif a == "x":
        rotated_x = x*np.cos(angle_x*np.pi/180)-y*np.sin(angle_x*np.pi/180)
        if rotated_x > 0:
            angle = angle_x
        elif rotated_x < 0:
            angle = 180+angle_x
        else:
            angle = angle_x
    else:
        print("ERROR: Wrong axis specified. Needs to be \"x\" or \"y\"!")
        exit()
            
    for atom in carbons:
        rot_xcoord=atom.position[0]*np.cos(angle*np.pi/180)-atom.position[1]*np.sin(angle*np.pi/180)
        rot_ycoord=atom.position[0]*np.sin(angle*np.pi/180)+atom.position[1]*np.cos(angle*np.pi/180)
        atom.position[0]=round(rot_xcoord, 12)
        atom.position[1]=round(rot_ycoord, 12)

    distances_opt = []
    color_list = []
    for [e1,e2] in edges:
        color = [0, 255, 255]
        distance = carbons.get_distance(e1, e2)
        distances_opt.append((distance-d_opt)**2)
        if distance < d_opt:
            factor = 2
            digit = 2
        if distance > d_opt:
            factor = 1
            digit = 1
        if distance > d_opt+0.1:
            c = 0
            print("WARNING: One of the bonds is longer than \"d_opt + 10 pm\"!")
        elif distance < d_opt-0.05:
            c = 0
            print("WARNING: One of the bonds is shorter than \"d_opt - 5 pm\"!")
        else:
            c = 255 - np.abs(distance-d_opt)*2550*factor
        color[digit] = c
        color_list.append(np.array(color)/255)

    O_HOMA = 1 - norm/len(distances_opt) * sum(distances_opt)
    print("Overall HOMA value:", np.round(O_HOMA,2))

    HOMA_rings = []
    ring_color = []
    for r in SSSR:
        d_list = []
        for i in r:
            for j in r:
                if [i,j] in edges:
                    pos = edges.index([i,j])
                    d_list.append(distances_opt[pos])
        HOMA = 1 - norm/len(d_list) * sum(d_list)
        HOMA_rings.append(HOMA)
        if HOMA <= 0:
            if HOMA <= -1:
                color = [255, 0, 255]
                print("WARNING: One of the rings has a HOMA value smaller than \"-1.00\"!")
            else:
                color = [255, 255, np.abs(HOMA*255)]
        else:
            color = [255, 255-np.abs(HOMA*255), 0]
        ring_color.append(np.array(color)/255)
        
    Y_min = min([i[1] for i in carbons.get_positions()])
    Y_max = max([i[1] for i in carbons.get_positions()])
    X_min = min([i[0] for i in carbons.get_positions()])
    X_max = max([i[0] for i in carbons.get_positions()])

    text_kwargs = dict(ha='center', va='center', fontsize=16, color='black')
    if not pbc:
        f=0.6
        plt.figure(figsize=[f*(X_max-X_min+0.8),f*(Y_max-Y_min+0.8)], dpi=300)
        plt.xlim(X_min-0.4,X_max+0.4)
        plt.ylim(Y_min-0.4,Y_max+0.4)
    else:
        f=0.5
        plt.figure(figsize=[f*(cell[0][0]+0.8),f*(cell[1][1]+0.8)], dpi=300)
        mid_point_X = (X_max+X_min)/2
        mid_point_Y = (Y_max+Y_min)/2
        plt.xlim(mid_point_X-cell[0][0]/2, mid_point_X+cell[0][0]/2)
        plt.ylim(mid_point_Y-cell[1][1]/2, mid_point_Y+cell[1][1]/2)
    plt.gca().set_aspect("equal")
    plt.axis('off')
    #plt.plot([-2,-2],[-2, 2], color="red", zorder=1000) # lines for testing
    #plt.plot([-2,2],[-2, -2], color="red", zorder=1000)
    for atom in carbons:
        if atom.symbol == "C":
            plt.plot(atom.position[0], atom.position[1], "o", color="black", zorder=100, ms=size)
        elif atom.symbol == "O":
            plt.plot(atom.position[0], atom.position[1], "o", color="red", zorder=100, ms=size)
        elif atom.symbol == "H":
            plt.plot(atom.position[0], atom.position[1], "o", color="grey", zorder=100, ms=size)
        else:
            plt.plot(atom.position[0], atom.position[1], "o", color="deeppink", zorder=100, ms=size)
    for pos,[e1,e2] in enumerate(edges):
        plt.plot([carbons[e1].position[0],carbons[e2].position[0]],[carbons[e1].position[1],carbons[e2].position[1]], color=color_list[pos], lw=2/3*size, zorder=50)
    for pos,r in enumerate(SSSR):
        plt.fill([carbons[i].position[0] for i in r], [carbons[i].position[1] for i in r], color=ring_color[pos], zorder=10)
        if not no_values:
            X_list = []
            Y_list = []
            for i in r:
                X_list.append(carbons[i].position[0])
                Y_list.append(carbons[i].position[1])
            middle_X = np.mean(X_list)
            middle_Y = np.mean(Y_list)
            plt.text(middle_X, middle_Y, '{:.2f}'.format(np.round(HOMA_rings[pos],2)), **text_kwargs, zorder=25)
    plt.savefig(filename, transparent=True)
    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate and plot HOMA values from atomic coordinates')
    parser.add_argument('Coordinates',type=str,help='File from which to load atomic coordinates')
    parser.add_argument('--file', help='Filename which is plotted', default='Coordinates.svg')
    parser.add_argument('--C1', help='Carbon atom which is centered', default=3)
    parser.add_argument('--C2', help='Carbon atom which is aligned', default=4)
    parser.add_argument('--axis', help='Axis on which two carbons atoms are aligned (\"x\"->horizaontal or \"y\"->vertical)', default="y")
    parser.add_argument('--d_opt', help='Optimal Benzene bond length', type=float, default=1.398)
    parser.add_argument('--norm', help='Normalization constant so that HOMA of 1,3,5-Cyclohexatriene is 0', type=float, default=362.9)
    parser.add_argument('--pbc_cutoff', help='Cutoff, for which carbon atoms outside of pbc cell are considered (increase if not all rings were used)', type=float, default=2.8)
    parser.add_argument('--pbc', help='Use periodic boundary conditions', action='store_true')
    parser.add_argument('--no_values', help='Plot no HOMA values inside rings', action='store_true')
    parser.add_argument('--rings', help='Manually specify rings', nargs='+', action='append', default=[]),
    parser.add_argument('--max_ring_len', help='Maximum length of ring that is considered (may HEAVILY reduce time for large molecules)', type=int, default=8)
    parser.add_argument('--no_of_cyc_combs', help='Maximum number of cycle combinations to obtain SSSR (may reduce time for large molecules, small number if max_ring_len is set)', type=int, default=4)
    parser.add_argument('--atom_types', help='Manually specify atom types (Note: optimal CC bond length is applied to CX bond!)', nargs='+', action='store', default=["C"])
    args = parser.parse_args()
    #start_time = timeit.default_timer()
    main(args.Coordinates, args.file, args.C1, args.C2, args.axis, args.d_opt, args.norm, args.rings, args.pbc_cutoff, args.max_ring_len, args.no_of_cyc_combs, args.atom_types, args.pbc, args.no_values)
    #stop_time = timeit.default_timer()
    #print('Time: ', np.round(stop_time - start_time,2), "s")  
