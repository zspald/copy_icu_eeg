#!/bin/bash
# A parallel bash scripts that allows the user to generate predictions for EEG data using the rf models
username=""
password=""
threshold=0.45 #0.45
length=60
# declare -a PATIENTS=(ICUDataRedux_0054 ICUDataRedux_0065 ICUDataRedux_0067 ICUDataRedux_0068)
# declare -a PATIENTS=(CNT685)
declare -a PATIENTS=(
				# CNT684 
				# CNT685 CNT687 CNT688 
				# CNT689 CNT690 CNT691 CNT692 
				# CNT694 CNT695 
				CNT698 CNT700 
				CNT701 CNT702 CNT705 CNT706 
				CNT708 CNT710 CNT711 CNT713 
				# CNT715 
				CNT720 
				# CNT723
				CNT724 CNT725 
				# CNT726 
				CNT729 CNT730 
				CNT731 CNT732 CNT733 CNT734 
				CNT737 CNT740 CNT741 CNT742 
				CNT743 CNT748 CNT750 CNT757 
				CNT758 CNT765 CNT773 CNT774 
				CNT775 CNT776 CNT778 CNT782
				CNT929 ICUDataRedux_0003 ICUDataRedux_0004 
				#### ICUDataRedux_0006 
				ICUDataRedux_0023 ICUDataRedux_0026	ICUDataRedux_0027 ICUDataRedux_0028 
				ICUDataRedux_0029 ICUDataRedux_0030 ICUDataRedux_0033 ICUDataRedux_0034 
				ICUDataRedux_0035
				### ICUDataRedux_0036
				ICUDataRedux_0040 ICUDataRedux_0042
				ICUDataRedux_0043 ICUDataRedux_0044	ICUDataRedux_0045 ICUDataRedux_0047
				ICUDataRedux_0048 ICUDataRedux_0049 ICUDataRedux_0050 ICUDataRedux_0054 
				ICUDataRedux_0060 ICUDataRedux_0061 ICUDataRedux_0062 ICUDataRedux_0063
				ICUDataRedux_0064 ICUDataRedux_0065	ICUDataRedux_0066 ICUDataRedux_0067 
				ICUDataRedux_0068
				#### ICUDataRedux_0069 
				ICUDataRedux_0072 ICUDataRedux_0073 
				ICUDataRedux_0074 ICUDataRedux_0078 
				### ICUDataRedux_0082 
				ICUDataRedux_0083
				ICUDataRedux_0084 ICUDataRedux_0085 ICUDataRedux_0086 ICUDataRedux_0087 
				ICUDataRedux_0089 ICUDataRedux_0090) 
				### ICUDataRedux_0091)
# declare -a PATIENTS=(
# 				# CNT710 CNT743 CNT929
# 				# ICUDataRedux_0026 ICUDataRedux_0030 ICUDataRedux_0040
# 				# ICUDataRedux_0045 ICUDataRedux_0050 ICUDataRedux_0062
# 				# ICUDataRedux_0066 ICUDataRedux_0073 ICUDataRedux_0084
# 				# ICUDataRedux_0089			
# 				)

# declare -a PATIENTS=(
# 				CNT685 ICUDataRedux_0060 
# 				# ICUDataRedux_0061 ICUDataRedux_0062
# )


num_pts=${#PATIENTS[@]}
num_par=3
# num_par=4
# echo $length

pos=0
for (( i=0; i<${num_pts}; i+=$num_par));
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
				python run_cnn_detector.py -u $username -p $password -id $pt -t $threshold -l $length -pos $pos&
				((pos+=1))
			done
			wait
			pos=0
	done
# 	echo $patient
	# python run_cnn_detector.py -u $username -p $password -id $pt -t $threshold -l $length &
# done
wait

