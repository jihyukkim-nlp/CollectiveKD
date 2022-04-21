#!/bin/bash
org_file_path=$1
split_dir=$2
n_org_lines=$(wc -l ${org_file_path} | awk '{ print $1 }') 

if [ -d ${split_dir} ];then
    n_split_lines_accum=0
    for each_file_path in $(ls ${split_dir});do
        _add=$(wc -l ${split_dir}/${each_file_path} | awk '{ print $1 }') 
        n_split_lines_accum=$(expr ${n_split_lines_accum} + ${_add})
    done
    if [ ! ${n_split_lines_accum} -eq ${n_org_lines} ];then
        echo "# of lines in original file and split files are different (org ${n_org_lines} != split ${n_split_lines_accum})"
        echo "Exit!"
        return 100
    else
        return 0
    fi
else
    n_oth_lines=$(wc -l ${split_dir} | awk '{ print $1 }') 
    if [ ! ${n_oth_lines} -eq ${n_org_lines} ];then
        echo "# of lines in file1 and fil2 are different (file1 ${n_org_lines} != file2 ${n_oth_lines})"
        echo "Exit!"
        return 100
    else
        return 0
    fi
fi