#!/bin/bash
# A parallel bash scripts that allows the user to pre-process the EEG data from IEEG and save for training
username=""
password=""
num_iter=20
batch_size=300 #1000
start=0
length=3
filter=1
eeg=1
norm=1
bipolar=0
random_forest=0
pool=0
ref_and_bip=0
deriv=0
# declare -a PATIENTS=(ICUDataRedux_0054 ICUDataRedux_0065 ICUDataRedux_0067 ICUDataRedux_0068 
# 					ICUDataRedux_0072 ICUDataRedux_0082 ICUDataRedux_0083)
declare -a PATIENTS=(
				# CNT684 
				CNT685 CNT687
				CNT688 CNT689 CNT690 CNT691
				CNT692 CNT694 CNT695 CNT698 
				CNT700 CNT701 CNT702 CNT705
				CNT706 CNT708 CNT710 CNT711 
				CNT713 CNT715 CNT720 CNT723
				CNT724 CNT725 CNT726 CNT729 
				CNT730 CNT731 CNT732 CNT733
				CNT734 CNT737 CNT740 CNT741		
				CNT742 CNT743 CNT748 CNT750 
				CNT757 CNT758 CNT765 
				CNT773 CNT774 CNT775 CNT776
				CNT778 CNT782 CNT929 ICUDataRedux_0003 
				ICUDataRedux_0004 ICUDataRedux_0006 ICUDataRedux_0023 ICUDataRedux_0026
				ICUDataRedux_0027 ICUDataRedux_0028 ICUDataRedux_0029 ICUDataRedux_0030 
				ICUDataRedux_0033 ICUDataRedux_0034 ICUDataRedux_0035 ICUDataRedux_0036
				ICUDataRedux_0040 ICUDataRedux_0042 ICUDataRedux_0043 ICUDataRedux_0044
				ICUDataRedux_0045 ICUDataRedux_0047 ICUDataRedux_0048 ICUDataRedux_0049 
				ICUDataRedux_0050 ICUDataRedux_0054 ICUDataRedux_0060 ICUDataRedux_0061 
				ICUDataRedux_0062 ICUDataRedux_0063 ICUDataRedux_0064 ICUDataRedux_0065
				ICUDataRedux_0066 ICUDataRedux_0067 ICUDataRedux_0068 ICUDataRedux_0069 
				ICUDataRedux_0072 ICUDataRedux_0073 ICUDataRedux_0074 ICUDataRedux_0078
				ICUDataRedux_0082 ICUDataRedux_0083 ICUDataRedux_0084 ICUDataRedux_0085 
				ICUDataRedux_0086 ICUDataRedux_0087 ICUDataRedux_0089 ICUDataRedux_0090 
				ICUDataRedux_0091)
# declare -a PATIENTS=(CNT684)
num_pts=${#PATIENTS[@]}
num_par=4
# echo $length

for (( i=0; i<${num_pts}; i+=4));
	do
		temp=()
		for (( j=0; j<$num_par; j++));
			do
				ind=$(($i + $j))
				# echo $ind
				if [ $ind -lt ${num_pts} ]; then
					temp+=(${PATIENTS[$ind]})
				fi
			done

		for pt in ${temp[@]};
			do
				# echo $pt
				python run_processor.py -u $username -p $password -id $pt -n $num_iter -b $batch_size -s $start -l $length -f $filter -eo $eeg -no $norm -bi $bipolar -rf $random_forest -po $pool -rb $ref_and_bip -df $deriv &
			done
			wait
		# echo next
	done

# for patient in ${PATIENTS[@]} 
# do
# 	python run_processor.py -u $username -p $password -id $patient -n $num_iter -b $batch_size -s $start -l $length -f $filter -eo $eeg -no $norm -bi $bipolar -rf $random_forest -po $pool -rb $ref_and_bip -df $deriv &
# done
wait

