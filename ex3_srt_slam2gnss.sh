#!/bin/bash
source /home/dongdong/1sorftware/1work/yes/etc/profile.d/conda.sh #激活conda环境




# data_dir_slamout="/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/300_gps/"
# slam_txt="${data_dir_slamout}/slam_out/1_1_frame_slam_enu.txt"   # 输入图像slamd定位位姿结果
# gps_intPionts="${data_dir_slamout}/slam_config/GNSS_config.yaml" #输入图像gnss参考点输入文件


# data_dir_loctae="/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/400_500_gps/" # 
# gps_txt="${data_dir_loctae}/slam_config/gnss.txt" # 待定位图像GNSS数据txt (时间戳 经纬高 空格分开)
# slam2gnss_SRt_xml="${data_dir_loctae}/slam_out/eavl_srt_slam2gnss.yaml" # 输出计算的srt关系保存路径


data_dir_slamout="/home/dongdong/2project/0data/RTK/nwpu/3_300_500_250/300_gps/"
slam_txt="/home/dongdong/2project/0data/RTK/nwpu/3_300_500_250/300_gps/slam_out/1_1_frame_slam_enu.txt"   # 输入图像slamd定位位姿结果
# slam_txt="/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/openvslam结果/批次2/500_450自己重定位开建图.txt"   # 输入图像slamd定位位姿结果
gps_intPionts="${data_dir_slamout}/slam_config/GNSS_config.yaml" #输入图像gnss参考点输入文件


data_dir_loctae="/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/500_450_gps/" # 
gps_txt="${data_dir_loctae}/slam_config/gnss.txt" # 待定位图像GNSS数据txt (时间戳 经纬高 空格分开)
slam2gnss_SRt_xml="${data_dir_loctae}/slam_out/eavl_srt_slam2gnss.yaml" # 输出计算的srt关系保存路径



# pip install geographiclib
conda activate gaussian_splatting
python ex3_srt_slam2gnss.py --slam_txt $slam_txt \
                          --gps_intPionts $gps_intPionts \
                          --gps_txt $gps_txt \
                          --slam2gnss_SRt_xml $slam2gnss_SRt_xml