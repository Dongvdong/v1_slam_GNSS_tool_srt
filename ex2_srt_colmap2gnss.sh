#!/bin/bash
source /home/dongdong/1sorftware/1work/yes/etc/profile.d/conda.sh #激活conda环境




# data_dir="/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/300_gps/"
# colmap_images_txt="${data_dir}/sparse/0/images.txt"   # 输入图像colmap重建位姿结果
# gps_txt="${data_dir}/slam_config/gnss.txt" # 输入图像GNSS数据txt (时间戳 经纬高 空格分开)
# gps_intPionts="${data_dir}/slam_config/GNSS_config.yaml" #输入图像gnss参考点输入文件
# colmap2gnss_SRt_xml="${data_dir}/sparse/srt_colmap2gnss.yaml" #输出计算的srt关系保存路径

data_dir="/home/dongdong/2project/0data/RTK/nwpu/3_300_500_250/300_gps"
colmap_images_txt="${data_dir}/sparse/0/images.txt"   # 输入图像colmap重建位姿结果
gps_txt="${data_dir}/slam_config/gnss.txt" # 输入图像GNSS数据txt (时间戳 经纬高 空格分开)
gps_intPionts="${data_dir}/slam_config/GNSS_config.yaml" #输入图像gnss参考点输入文件
colmap2gnss_SRt_xml="${data_dir}/sparse/srt_colmap2gnss.yaml" #输出计算的srt关系保存路径


# pip install geographiclib
conda activate gaussian_splatting
python ex2_srt_colmap2gnss.py --colmap_images_txt $colmap_images_txt \
                          --gps_intPionts $gps_intPionts \
                          --gps_txt $gps_txt \
                          --colmap2gnss_SRt_xml $colmap2gnss_SRt_xml