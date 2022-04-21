#!/bin/bash
id=${1}
dest=${2}
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${id}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${id}" -o ${dest}
rm ./cookie