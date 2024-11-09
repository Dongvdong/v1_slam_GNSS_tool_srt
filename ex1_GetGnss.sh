#!/bin/bash
source /home/dongdong/1sorftware/1work/yes/etc/profile.d/conda.sh

data_dir="/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/500_450_gps/"
input_img_path="${data_dir}/原始数据/images/"   # 输入图像colmap重建位姿结果
Save_GPS_txt="${data_dir}/gnss.txt" # 输入图像GNSS数据txt (时间戳 经纬高 空格分开)

conda activate gaussian_splatting
python ex1_GetGnss.py --input_img_path $input_img_path --Save_GPS_txt $Save_GPS_txt