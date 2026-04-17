#!/bin/bash

# The only thing you need to change: The path to the POTCARs!
POTDIR='/PUT/PATH/TO/POTCARS/HERE/'

# Some general variables 
myPWD=$(pwd)
decision='0'

# Function for how to use the script
function usage
{
    echo " "
    echo "* DESCRIPTION :  This script reads the POSCAR file and creates the POTCAR file of corresponding elements in order."
    echo " "
    echo "* USAGE: $(basename "$0") [-h|--help|-rnspdbcSHgG]"
    echo "  -h | --help :  For help."
    echo "  -r          :  If you want the recommended default potentials (see IMPORTANT)"
    echo "  -n          :  If you want no extension of the POTCAR (minimal valence)"
    echo "  -s          :  If you want the _sv extension of the POTCAR (semicore p and s states in valence)"
    echo "  -p          :  If you want the _pv extension of the POTCAR (semicore p states in valence)"
    echo "  -d          :  If you want the _d extension of the POTCAR (semicore d states in valence)"
    echo "  -b          :  If you want the _2 extension of the POTCAR (some 4f electrons of Lanthanides are frozen)"
    echo "  -c          :  If you want the _3 extension of the POTCAR (some 4f electrons of Lanthanides are frozen)"
    echo "  -S          :  If you want the _s extension of the POTCAR (soft potential)"
    echo "  -H          :  If you want the _h extension of the POTCAR (hard potential)"
    echo "  -g          :  If you want the _GW extension of the POTCAR (opt. for unoccupied states)"
    echo "  -G          :  If you want the _sv_GW extension of the POTCAR (see _sv & _GW)"
    echo " "
    echo "* IMPORTANT   :  It uses the following extensions as the recommended default potentials (-r):"
    echo "  None        :  H, He, Be, B, C, N, O, F, Ne, Mg, Al, Si, P, S, Cl, Ar, Co,"
    echo "                 Ni, Cu, Zn, As, Se, Br, Kr, Pd, Ag, Cd, Sb, Te, I, Xe, La, Ce,"
    echo "                 Re, Os, Ir, Pt, Au, Hg, At, Rn, Ac, Th, Pa, U, Np, Pu, Am, Cm"
    echo "  sv          :  Li, K, Ca, Sc, Ti, V, Rb, Sr, Y, Zr, Nb, Mo, Cs, Ba, W, Fr, Ra"
    echo "  pv          :  Na, Cr, Mn, Tc, Ru, Rh, Hf, Ta"
    echo "  d           :  Ga, Ge, In, Sn, Tl, Pb, Bi, Po"
    echo "  2           :  Eu, Yb"
    echo "  3           :  Pr, Nd, Pm, Sm, Gd, Tb, Dy, Ho, Er, Tm, Lu"
    echo " "
    echo "* NOTE        :  To create the POTCAR you must need the POSCAR file. You first have to move to"
    echo "                 the folder that contains the POSCAR. It can't create POTCAR from external path."
    echo " "
    echo "---------------  Enjoy. Have a good day.  ---------------"
    echo " "
    exit 0
}

# Function for what changes when the options are read in
function readoptions
{
    OPTIND=1
    while getopts :rnspdbcSHgG options; do
        case $options in
            r) decision='1' ;;
            n) extension='' ;;
            s) extension='_sv' ;;
            p) extension='_pv' ;;
            d) extension='_d' ;;
            b) extension='_2' ;;
            c) extension='_3' ;;
            S) extension='_s' ;;
            H) extension='_h' ;;
            g) extension='_GW' ;;
            G) extension='_sv_GW' ;;
            \?) printf "\n\e[38;5;1m* Error: Invalid option %s. Use -h for help. \e[0m\n\n" "$1"; exit 1 ;;
        esac
    done
}

# Check if help is needed, if there are no or too many options given, and change variables according to options
if [ -n "$2" ]; then
    printf "\n\e[38;5;1m* Error: Only one option allowed at a time! Use -h for help. \e[0m\n\n"
    exit 1
elif [ -n "$1" ]; then
    if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        usage
    elif [ "$1" == "-" ]; then
        printf "\n\e[38;5;1m* Error: Invalid option '-'. Use -h for help. \e[0m\n\n"
        exit 1
    elif [[ "$1" == -* ]]; then
        if [[ ${#1} -gt 2 ]]; then
            printf "\n\e[38;5;1m* Error: Only one option allowed at a time! You passed combined options %s. Use -h for help. \e[0m\n\n" "$1"
            exit 1
        else
            readoptions "$@"
        fi
    else
        printf "\n\e[38;5;1m* Error: Invalid option %s. Use -h for help. \e[0m\n\n" "$1"
        exit 1
    fi
else
    printf "\n\e[38;5;1m* Error: You must specify one option! Use -h for help. \e[0m\n\n"
    exit 1
fi

# Definition of recommended default options
declare -A element_group=(
    [H]="" [He]="" [Be]="" [B]="" [C]="" [N]="" [O]="" [F]="" [Ne]="" [Mg]="" [Al]="" [Si]="" [P]="" [S]="" [Cl]="" [Ar]="" [Co]="" [Ni]="" [Cu]="" [Zn]="" [As]="" [Se]="" [Br]="" [Kr]=""
    [Pd]="" [Ag]="" [Cd]="" [Sb]="" [Te]="" [I]="" [Xe]="" [La]="" [Ce]="" [Re]="" [Os]="" [Ir]="" [Pt]="" [Au]="" [Hg]="" [At]="" [Rn]="" [Ac]="" [Th]="" [Pa]="" [U]="" [Np]="" [Pu]="" [Am]="" [Cm]=""
    [Li]="_sv" [K]="_sv" [Ca]="_sv" [Sc]="_sv" [Ti]="_sv" [V]="_sv" [Rb]="_sv" [Sr]="_sv" [Y]="_sv" [Zr]="_sv" [Nb]="_sv" [Mo]="_sv" [Cs]="_sv" [Ba]="_sv" [W]="_sv" [Fr]="_sv" [Ra]="_sv"
    [Na]="_pv" [Cr]="_pv" [Mn]="_pv" [Tc]="_pv" [Ru]="_pv" [Rh]="_pv" [Hf]="_pv" [Ta]="_pv"
    [Ga]="_d" [Ge]="_d" [In]="_d" [Sn]="_d" [Tl]="_d" [Pb]="_d" [Bi]="_d" [Po]="_d"
    [Eu]="_2" [Yb]="_2"
    [Pr]="_3" [Nd]="_3" [Pm]="_3" [Sm]="_3" [Gd]="_3" [Tb]="_3" [Dy]="_3" [Ho]="_3" [Er]="_3" [Tm]="_3" [Lu]="_3"
)

# Execution of creating the POTCAR
if [ -f POSCAR ]; then # Check if POSCAR exists
    if [ -s POTCAR ]; then # Check if old POTCAR already exists
        printf "\n\e[38;5;9;4m* Warning:\e[0m Hi %s, you already have an old POTCAR. I am deleting it and creating the new one.\n\n" "$(whoami)"
        rm POTCAR
    fi
    ATOMS=$(sed -n '6p' POSCAR) && nA=$(wc -w <<< "$ATOMS") # Get elements
    NUMBERS=$(sed -n '7p' POSCAR) && nN=$(wc -w <<< "$NUMBERS")  # Get number of atoms
    if [ "$nA" -ne "$nN" ]; then # Check if number of elements in line 6 and 7 match
        printf "\n\e[38;5;1m* Error: Number of elements in line 6 (elements) and line 7 (count) do not match: %s vs. %s \e[0m \n\n" "$nA" "$nN"
        exit 1
    fi
    echo "POTCAR for these elements will be created (in order): $ATOMS"
    for element in $ATOMS
    do
        if [ "$decision" == "1" ]; then # Check if recommended default potentials should be used
            extension="${element_group[$element]}" # Assign extension for each element
        fi
        if [ ! -d "$POTDIR$element$extension" ]; then # Check if element with extension actually exists 
            printf "\n\e[38;5;1m* Error: %s does not exist.\e[0m \n\n" "$(printf '%q' "$POTDIR$element$extension")"
            exit 1
        fi
        cat "$POTDIR$element$extension/POTCAR" >> "$myPWD/POTCAR" # Important part: attach the POTCARs to each other in order
    done
else
    printf "\n\e[38;5;1m* Error: No POSCAR file here! POSCAR file is mandatory. Please move to the file containing the POSCAR. \e[0m\n\n"
    exit 1
fi

