#!/bin/bash
# Uses chgcar2cube.py and VMD to create a visualization of the magnetization density
# Usage: visualize-magnetization CHGCAR density.cube
# will create density.vmd and density_mag.vmd for use with VMD alongside two cube files.
set -e

if [ -f "$2" ]; then
   echo "$2 exists, skipping CHGCAR conversion!"
else
   echo "Converting CHGCAR to $2."
   dir=$(dirname "$0")
   "$dir"/chgcar2cube.py "$1" -o "${2%.cube}" -v --volume
   echo "... done."
fi

vmdfile=${2%.cube}.vmd
if [ -f "$vmdfile" ]; then
   echo "$vmdfile exists, skipping VMD file creation!"
else
   {
   echo "Writing $vmdfile"
   echo "mol new $2 type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all"
   echo "mol representation CPK 1.000000 0.000000 32.000000 22.000000"
   echo "mol addrep top"
   echo "mol representation DynamicBonds 2.100000 0.100000 22.000000"
   echo "mol addrep top"
   echo "mol representation Isosurface 0.05000 0 0 1 1 1"
   echo "mol color ColorID 1"
   echo "mol addrep top"
   echo "mol representation Isosurface -0.05000 0 0 1 1 1"
   echo "mol color ColorID 0"
   echo "mol addrep top"
   } >> "$vmdfile"
   #echo "" >> $vmdfile
   if [ -f "${vmdfile%.vmd}"_mag.vmd ]; then
      echo "${vmdfile%.vmd}_mag.vmd exists, skipping VMD file creation!"
   else
      cp "$vmdfile" "${vmdfile%.vmd}"_mag.vmd
      string="s/${2%.cube}/${2%.cube}_mag/g"
      sed -i "$string" "${vmdfile%.vmd}"_mag.vmd
   fi
fi

echo -e "Run\nvmd -e $vmdfile\nor\nvmd -e ${vmdfile%.vmd}_mag.vmd"