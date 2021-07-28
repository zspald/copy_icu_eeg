#!/bin/bash
# A parallel bash scripts that allows the user to generate predictions for EEG data using the rf models
username="zspald"
password="FimbleWinter5994@"
threshold=0.45
length=60
bipolar=1
pool=0
# declare -a PATIENTS=(ICUDataRedux_0054 ICUDataRedux_0065 ICUDataRedux_0067 ICUDataRedux_0068 
# 					ICUDataRedux_0072 ICUDataRedux_0082 ICUDataRedux_0083)
declare -a PATIENTS=(
				# CNT684 CNT685 CNT687 CNT688 
				# CNT689 CNT690 CNT691 CNT692)
				# CNT694 CNT695 CNT698 CNT700 
				# CNT701 CNT702 CNT705 CNT706)
				# ICUDataRedux_0054 ICUDataRedux_0061 ICUDataRedux_0062 ICUDataRedux_0063 
				# ICUDataRedux_0064 ICUDataRedux_0065 ICUDataRedux_0068 ICUDataRedux_0069) 
				# ICUDataRedux_0072 ICUDataRedux_0073 ICUDataRedux_0074 ICUDataRedux_0078 
				# ICUDataRedux_0082 ICUDataRedux_0083 ICUDataRedux_0084 ICUDataRedux_0085) 
				ICUDataRedux_0086 ICUDataRedux_0087 ICUDataRedux_0089 ICUDataRedux_0090 
				ICUDataRedux_0091)
# declare -a PATIENTS=(CNT684)
for patient in ${PATIENTS[@]} 
do
	python rf_detector.py -u $username -p $password -id $patient -b $bipolar -po $pool -t $threshold -l $length &
done
wait

