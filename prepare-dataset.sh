#!/bin/bash

set -euo pipefail

# Check for at least two arguments
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <dataset-name> <class1> [<class2> ...]"
  exit 1
fi

# First argument is the dataset name
dataset="$1"
out_dir=datasets/${dataset}
yaml_file=datasets/${dataset}.yaml

# Shift arguments so $@ contains only class names
shift

# Remaining arguments are the class list
classes=("$@")

# Ensure output directory does not exist
if [ -e "${out_dir}" ]; then
  echo "Error: output directory '${out_dir}' already exists"
  exit 1
fi

# Ensure yaml file does not exist
if [ -e "${yaml_file}" ]; then
  echo "Error: yaml file '${yaml_file}' already exists"
  exit 1
fi

# Create output directory
echo "Preparing dataset directory: ${out_dir}"
mkdir -p ${out_dir}

# Copy annotations over as is
echo "Moving annotations..."
mkdir -p ${out_dir}/annotations
cp OID/csv_folder/* ${out_dir}/annotations

# Copy labels and images for each class and subset
echo "Moving images and labels..."
for subset in train test validation; do
  echo "- subset ${subset}"
  mkdir -p ${out_dir}/{images,labels}/${subset}
  for i in "${!classes[@]}"; do
    class=${classes[$i]}
    echo "  - class ${class}"
    echo "    - copying images"
    cp OID/Dataset/${subset}/${class}/*.jpg ${out_dir}/images/${subset}
    echo "    - copying labels"
    cp OID/Dataset/${subset}/${class}/*.txt ${out_dir}/labels/${subset}
  done
done

# Write dataset yaml file
echo "Writing yaml file..."
cat > ${yaml_file} <<EOF
path: ../${out_dir}
train: images/train
test: images/test
val: images/validation

# Classes
names:
EOF

# Complete with individual class names
for i in "${!classes[@]}"; do
  echo "  ${i}: ${classes[$i]}" >> ${yaml_file}
done

echo "Done!"
