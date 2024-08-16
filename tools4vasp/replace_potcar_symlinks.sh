#!/bin/bash
# 
set -e
set -o pipefail
shopt -s globstar



# print help
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
   echo "Usage: $0"
   echo "Search for all POTCAR files in all subdirectories and replace them with symlinks to the corresponding POTCAR in the parent directory."
   exit 0
fi


# find all POTCAR files in all subdirectories
mapfile -t potcar_files < <(find . -mindepth 2 -type f -name "*POTCAR*")
mapfile -t ref_potcar_files < <(find . -mindepth 1 -maxdepth 1 -type f -name "*POTCAR*")
mapfile -t ref_md5sums < <(md5sum "${ref_potcar_files[@]}" | cut -d' ' -f1)
echo "Found the following reference POTCAR files:"
echo "${ref_potcar_files[@]}"
echo "With the following md5sums:"
echo "${ref_md5sums[@]}"
for file in "${potcar_files[@]}"; do
    # find corresponding POTCAR in parent directory
    md5=$(md5sum "$file"| cut -d' ' -f1)
    # check if md5 is in ref_md5sums
    if [[ ! "${ref_md5sums[*]}" =~ ${md5} ]]; then
        echo "No reference POTCAR found for $file skipping."
        continue
    fi
    # index of md5 in ref_md5sums
    index=""
    for i in "${!ref_md5sums[@]}"; do
        if [[ "${ref_md5sums[$i]}" = "${md5}" ]]; then
            index="${i}"
            break
        fi
    done
    ref_potcar=${ref_potcar_files[$index]}
    rel_path=$(realpath --relative-to="$(dirname "$file")" "$ref_potcar")
    echo "Found $file replacing with symlink to $ref_potcar with relative path $rel_path"
    # replace with symlink
    rm "$file"
    ln -s "$rel_path" "$file"
done

