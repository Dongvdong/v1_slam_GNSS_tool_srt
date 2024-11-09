#!/bin/bash
source /home/dongdong/1sorftware/1work/yes/etc/profile.d/conda.sh #激活conda环境




data_dir="data/ex3_testdata"
slam_txt="${data_dir}/trajectory.txt"   # 输入图像colmap重建位姿结果
gps_txt="${data_dir}/gnss.txt" # 输入图像GNSS数据txt (时间戳 经纬高 空格分开)
gps_intPionts="${data_dir}/GNSS_config.yaml" #输入图像gnss参考点输入文件
slam2gnss_SRt_xml="${data_dir}/srt_slam2gnss.yaml" #输出计算的srt关系保存路径

# pip install geographiclib
conda activate gaussian_splatting
python ex3_srt_slam2gnss.py --slam_txt $slam_txt \
                          --gps_intPionts $gps_intPionts \
                          --gps_txt $gps_txt \
                          --slam2gnss_SRt_xml $slam2gnss_SRt_xml