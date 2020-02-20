#!/bin/bash
PATIENTS=(RID0061 RID0062)
for patient in ${PATIENTS[@]}
do
	echo $patient 
	python test.py -pt $patient -u danieljkim0118 -p kjm39173917#
done