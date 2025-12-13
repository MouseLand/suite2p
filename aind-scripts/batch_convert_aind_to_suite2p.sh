#!/bin/bash
#
# Batch convert multiple AIND datasets to suite2p format
#
# Usage: ./batch_convert_aind_to_suite2p.sh <input_dir> <output_dir>
#
# This script will:
# - Find all dataset folders in input_dir
# - Convert each dataset to suite2p format
# - Skip datasets that have already been converted
# - Use the dataset folder name as the output dataset name

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    echo ""
    echo "Example:"
    echo "  $0 /s3-cache/suite2p-dev /output/suite2p_converted"
    echo ""
    echo "This will process all dataset folders in /s3-cache/suite2p-dev/"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONVERT_SCRIPT="$SCRIPT_DIR/convert_aind_to_suite2p.py"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Check if conversion script exists
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo -e "${RED}Error: Conversion script not found: $CONVERT_SCRIPT${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Batch AIND to Suite2p Conversion${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "Input directory:  ${INPUT_DIR}"
echo -e "Output directory: ${OUTPUT_DIR}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Find all potential dataset folders (containing VISp_X subfolders)
DATASETS=()
for folder in "$INPUT_DIR"/*; do
    if [ -d "$folder" ]; then
        # Check if folder contains VISp_X subfolders
        if ls "$folder"/VISp_* 1> /dev/null 2>&1; then
            DATASETS+=("$folder")
        fi
    fi
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No datasets found in $INPUT_DIR${NC}"
    echo "Looking for folders containing VISp_X subfolders"
    exit 0
fi

echo -e "${GREEN}Found ${#DATASETS[@]} dataset(s) to process${NC}"
echo ""

# Process each dataset
PROCESSED=0
SKIPPED=0
FAILED=0

for dataset_path in "${DATASETS[@]}"; do
    # Get dataset name (folder name)
    dataset_name=$(basename "$dataset_path")
    
    # Check if output already exists
    output_path="$OUTPUT_DIR/$dataset_name"
    
    echo -e "${BLUE}-----------------------------------------------------------${NC}"
    echo -e "Dataset: ${GREEN}$dataset_name${NC}"
    
    if [ -d "$output_path" ]; then
        echo -e "${YELLOW}  ⊘ Output already exists, skipping${NC}"
        echo -e "    $output_path"
        ((SKIPPED++))
        continue
    fi
    
    echo -e "  Processing..."
    echo -e "    Input:  $dataset_path"
    echo -e "    Output: $output_path"
    
    # Run conversion
    if python "$CONVERT_SCRIPT" \
        --input "$dataset_path" \
        --output "$OUTPUT_DIR" \
        --dataset-name "$dataset_name"; then
        echo -e "${GREEN}  ✓ Successfully converted${NC}"
        ((PROCESSED++))
    else
        echo -e "${RED}  ✗ Conversion failed${NC}"
        ((FAILED++))
    fi
    
    echo ""
done

# Print summary
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Batch Conversion Summary${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "Total datasets found: ${#DATASETS[@]}"
echo -e "${GREEN}Successfully processed: $PROCESSED${NC}"
echo -e "${YELLOW}Skipped (already exists): $SKIPPED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo -e "${BLUE}============================================================${NC}"

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
elif [ $PROCESSED -eq 0 ] && [ $SKIPPED -eq 0 ]; then
    exit 1
else
    exit 0
fi
