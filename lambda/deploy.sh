#!/bin/bash

# Define variables
LAYER_DIR="python"
ZIP_FILE="lambda_layer.zip"

# Remove old directories and zip file
rm -rf $LAYER_DIR $ZIP_FILE

# Create the python directory
mkdir -p $LAYER_DIR

# Install dependencies into the python directory
pip install -r requirements.txt -t $LAYER_DIR

# Zip the python directory to create 
zip -r $ZIP_FILE $LAYER_DIR

echo "Lambda Layer package created: $ZIP_FILE"