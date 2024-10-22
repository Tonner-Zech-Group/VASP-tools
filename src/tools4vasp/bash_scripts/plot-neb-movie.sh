#!/bin/bash
# Use VMD and plotNEB.py to create images for NEB curve presentation
#set -e

echo "Plotting Data..."
nImages=$(plotNEB --presentation --file NEB_presentations.png --plotall | tail -1 | cut -d" " -f1)
echo "Processing ${nImages} images."

if [[ ! -f "movie.vmd" ]]; then
   read -rp "VMD will now open, save the view as movie.vmd, then exit. Press enter to continue"
   vmd movie.xyz
fi

if tail -1 movie.vmd | grep -iq "quit"
then
   echo "movie.vmd already appended, skipping"
else
   echo "render options Tachyon '/usr/local/vmd/lib/vmd/tachyon_LINUXAMD64 -aasamples 12 %s -format TARGA'" >> movie.vmd
   for (( i=0; i<nImages; i++ ))
   do
      echo "animate goto ${i}" >> movie.vmd
      pi=$(printf "%02d" "$i")
      echo "render Tachyon NEB-${pi}.dat" >> movie.vmd
   done
   echo "quit" >> movie.vmd
fi

echo "Now plotting, give me time"
vmd -dispdev text -e movie.vmd

for (( i=0; i<nImages; i++ ))
do
   pi=$(printf "%02d" "$i")
   file="NEB-${pi}"
   echo "Rendering $file"
   sed -i 's/Resolution.*/Resolution 7664 4164/' "$file".dat
   tachyon  -aasamples 12 "$file".dat -format TARGA -o "$file".tga
   convert "$file".tga "$file".png
   rm "$file".tga
done

#pbc set { {21.9829320199999998   0.1410303600000000   0.0358512600000000} {11.1131279900000006  19.2103790399999994   0.0444921100000000} {11.0622655400000003   6.4760316500000004  18.0611593100000007}  } -all -namd -alignx
#vmd > pbc box -center unitcell
#vmd > animate next
#render TachyonInternal 0-0-0-0-0-0-0-1.xyz.tga
