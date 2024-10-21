import sys
import os

nc_file = sys.argv[1]
out_dat_file = os.path.basename(nc_file).replace('.nc','.dat')
out_txt_file = os.path.basename(nc_file).replace('.nc','.txt')
skip_detect = int(sys.argv[2])

detect_command = "DetectNodes \
            --in_data "+nc_file+" \
            --out "+out_dat_file+" \
            --timefilter \"6hr\" \
            --searchbymin sea_press \
            --closedcontourcmd \"sea_press,200.0,5.5,0\" \
            --mergedist 6.0 \
            --outputcmd \"sea_press,min,0;_VECMAG(u10,v10),max,2\""

stitch_command = "StitchNodes \
                --in "+out_dat_file+" \
                --out "+out_txt_file+" \
                --in_fmt \"lon,lat,slp,wind\" \
                --range 6.0 \
                --mintime \"54h\" \
                --maxgap \"18h\" \
                --threshold \"wind,>=,11.0,10\" \
                --out_file_format \"csv\""


if skip_detect != 1:
    print('Detect Node')
    print(detect_command)
    os.system(detect_command)

print('Stitch Node')
print(stitch_command)
os.system(stitch_command)
    