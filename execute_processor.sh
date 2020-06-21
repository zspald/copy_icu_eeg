#!/bin/bash
# A parallel bash scripts that allows the user to pre-process the EEG data from IEEG and save for training
username="INPUT IEEG USERNAME HERE"
password="INPUT IEEG PASSWORD HERE"
num_iter="INPUT NUMBER OF ITERATIONS"
batch_size="INPUT BATCH SIZE"
start="INPUT START TIME"
length="INPUT LENGTH OF EACH SEGMENT"
filter="INPUT WHETHER TO APPLY FILTER AS 0/1"
eeg="INPUT WHETHER TO ONLY USE EEG CHANNELS AS 0/1"
norm="INPUT WHETHER TO NORMALIZE THE DATA AS 0/1"
declare -a PATIENTS=("LIST ALL PATIENTS IN HERE")
for patient in "${PATIENTS[@]}"
do
	python run_processor.py -u "$username" -p "$password" -id "$patient" -n "$num_iter" -b "$batch_size" -s "$start" -l "$length" -f "$filter" -eo "$eeg" -no "$norm" &
done
wait

