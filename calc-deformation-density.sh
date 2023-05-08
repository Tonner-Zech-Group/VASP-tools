#!/bin/bash
# Three arguments: AB folder, A folder, B folder
set -e

function split_convert() {
   pwd
   if [ -f "$1".cube ]; then
      echo "$1.cube exists, skipping CHGCAR conversion!"
   else
      echo "Converting CHGCAR to $1.cube."
      chgcar2cube.py CHGCAR -o "$1" -v --volume
      echo "... done."
   fi
}

#check number of args
if [ "$#" -ne 3  ]; then
   echo "I need 3 arguments!"
   exit 1
fi

#loop over args
for var in "$@"
do
   if [ ! -d "$var" ]; then
      echo "$var is not a directory"
      exit 1
   fi
done


root=${PWD}

one=${1##*/}
if [ -z "$one" ]; then
   one="AB"
fi
echo "$one"
cd "$1"
pwd
split_convert "$one"
cd "$root"

two=${2##*/}
if [ -z "$two" ]; then
   two="A"
fi
echo "$two"
cd "$2"
split_convert "$two"
cd "$root"

three=${3##*/}
if [ -z "$three" ]; then
   three="B"
fi
echo "$three"
cd "$3"
split_convert "$three"
cd "$root"

echo "Calculating Total Deformation Density."
#A+B
sumCube.py -f "$2"/"$two".cube "$3"/"$three".cube -o tmp.cube
#AB-(A+B)
sumCube.py -f "$1"/"$one".cube tmp.cube -o deformation_density.cube --subtract
rm tmp.cube
echo "Calculating Magnetization Deformation Density."
#A+B
sumCube.py -f "$2"/"${two}"_mag.cube "$3"/"${three}"_mag.cube -o tmp.cube
#AB-(A+B)
sumCube.py -f "$1"/"${one}"_mag.cube tmp.cube -o deformation_density_mag.cube --subtract
rm tmp.cube

echo "Done!"
