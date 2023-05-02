#!/bin/bash

#from test numpy to validation figures!

in_dir=/scratch/08589/hvtran/predrnn-pytorch/checkpoints/2022_predrnn_3days/test_result/
out_dir=/scratch/08589/hvtran/PredRNN_Sandy/
out_dir_regrid=/scratch/08589/hvtran/PredRNN_Sandy/raw
start_date="2021-09-01-00"
skip_time=1
temp_file1="temp_output_step1.txt"
temp_file2="temp_output_step2.txt"

rm $temp_file1
rm $temp_file2

##### WIND ######
var_key="wind"
#Write numpy files to .nc files
python3 /work/08589/hvtran/ls6/ERA5_PredRNN-main/validation/WeatherBench/src/format_PredRNN_data-single.py $in_dir $out_dir $start_date $var_key > $temp_file1

##### SEA PRESS ######
var_key="sea_press"
#Write numpy files to .nc files
python3 /work/08589/hvtran/ls6/ERA5_PredRNN-main/validation/WeatherBench/src/format_PredRNN_data-single.py $in_dir $out_dir $start_date $var_key >> $temp_file1

##### PRECIPITATION ######
var_key="precip"
#Write numpy files to .nc files
python3 /work/08589/hvtran/ls6/ERA5_PredRNN-main/validation/WeatherBench/src/format_PredRNN_data-single.py $in_dir $out_dir $start_date $var_key >> $temp_file1

#Downscale from original resolution to 0.5 degree
while read -r line; do
    name="$line"
    new_name="${name/.nc/_05.nc}"
    echo $new_name >> $temp_file2
    python3 /work/08589/hvtran/ls6/ERA5_PredRNN-main/validation/WeatherBench/src/regrid.py --input_fns $name --output_dir $out_dir_regrid --ddeg_out 0.5 --reuse_weights 0 --custom_fn "${new_name##*/}"
done < "$temp_file1"

#rm $temp_file

#Calculate statistics
python3 /work/08589/hvtran/ls6/ERA5_PredRNN-main/validation/WeatherBench/src/validation.py $temp_file1 $start_date $skip_time /work/08589/hvtran/ls6/ERA5_PredRNN-main/validation/WeatherBench/src/results/2021/

