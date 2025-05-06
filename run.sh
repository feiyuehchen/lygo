#!/bin/bash

for i in {1..60}
do
  echo "Running iteration $i/60"
  python process_phonemes.py
done