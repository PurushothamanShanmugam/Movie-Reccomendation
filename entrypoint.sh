#!/bin/bash

echo "----------------------------------------"
echo "Movie Recommendation System Docker Start"
echo "----------------------------------------"

echo "Checking directories..."

mkdir -p data/processed
mkdir -p outputs/models
mkdir -p outputs/metrics
mkdir -p outputs/figures
mkdir -p outputs/recommendations

echo "Directories ready"

echo "----------------------------------------"
echo "Running recommendation pipeline"
echo "----------------------------------------"

python main.py

echo "----------------------------------------"
echo "Pipeline finished successfully"
echo "Outputs generated in outputs/ folder"
echo "----------------------------------------"