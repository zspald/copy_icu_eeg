#!/bin/bash
# A parallel bash scripts that allows the user to pre-process the EEG data from IEEG and save for training
username="zspald"
password="FimbleWinter5994@"
num_iter=20
batch_size=1000
start=0
length=1
filter=1
eeg=1
norm=1
bipolar=0
random_forest=1
pool=0
# declare -a PATIENTS=(CNT685 ICUDataRedux_0085)
declare -a PATIENTS=(
				# CNT684 CNT685 CNT687 CNT688 
				# CNT689 CNT690 CNT691 CNT692
				# CNT694 CNT695 CNT698 CNT700 
				# CNT701 CNT702 CNT705 CNT706)
				# ICUDataRedux_0054 ICUDataRedux_0060 ICUDataRedux_0061 ICUDataRedux_0062 
				# ICUDataRedux_0063 ICUDataRedux_0064 ICUDataRedux_0065 ICUDataRedux_0067
				# ICUDataRedux_0068 ICUDataRedux_0069 ICUDataRedux_0072 ICUDataRedux_0073 
				# ICUDataRedux_0074 ICUDataRedux_0078 ICUDataRedux_0082 ICUDataRedux_0083) 
				ICUDataRedux_0084 ICUDataRedux_0085 ICUDataRedux_0086 ICUDataRedux_0087
				ICUDataRedux_0089 ICUDataRedux_0090 ICUDataRedux_0091)
# declare -a PATIENTS=(CNT684 CNT684 CNT687 CNT688)
for patient in ${PATIENTS[@]} 
do
	python run_processor.py -u $username -p $password -id $patient -n $num_iter -b $batch_size -s $start -l $length -f $filter -eo $eeg -no $norm -bi $bipolar -rf $random_forest -po $pool &
done
wait

