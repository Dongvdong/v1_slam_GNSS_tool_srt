# v1_slam_GNSS_tool_srt

1 提取GNSS  

2 图像resize并且保存新的内参文件  

3 计算colmap sparse到gnss的srt转换关系  

4 计算slam txt位姿到gnss的srt转换关系和均方根误差  

5 画图  


更新2============
all1_getgps_resizeImg.py

'''
作用： 数据预处理
1-1 抽取GNSS保存txt
1-2 绘制GPS轨迹
1-3 统计GNSS信息，公里，面积等
1-4 读取相机参数，缩放相机参数，保存缩放后的相机参数
1-5 去畸变并保存缩放后的图像

输入：
1-1 input_images_dir: 图像文件夹路径
确保原图格式   
数据文件夹1/source_images/DJI_0001.JPG ... DJI_0099.JPG
数据文件夹2/source_images/DJI_0001.JPG ... DJI_0099.JPG
...
1-2 input_camconfig_path: 相机参数文件路径
数据文件夹1/slam_config/GNSS_config_distort.yaml
数据文件夹2/slam_config/GNSS_config_distort.yaml
...
1-3 WorkMode  该数据用来干什么
建图阶段和定位阶段根据字段  开启不同的建图和优化策略

输出
保存地址: 数据文件夹1/slam_config/
1-1 每张照片的 GNSS数据  gnss.txt
1-2 网页GNSS轨迹图 gps_trajectory_map.html
1-2 场景大小信息.txt
1-4 处理后的相机内参 GNSS_config.yaml
1-5 缩放畸变矫正的后的图像文件夹  数据文件夹1/images

'''
