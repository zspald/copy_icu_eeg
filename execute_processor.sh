#!/bin/bash
declare -a PATIENTS=("RID0061" "RID0062" "RID0063" "RID0064")
for patient in "${PATIENTS[@]}"
do
	python run_processor.py -u danieljkim0118 -p kjm39173917# -id $patient -n 5 -b 20 -s 0 -l 5 -f 1 -eo 1 -no 0 &
done
wait