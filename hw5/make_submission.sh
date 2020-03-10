#!/usr/bin/env bash
zip -r hw1.zip Problem_1 Problem_2 Problem_3 -x "Problem_1/hog_model/*" "Problem_1/__pycache__/*" "Problem_2/datasets/*" '*.npz' '*.npy'
