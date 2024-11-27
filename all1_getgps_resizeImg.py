import cv2
import yaml
import os
import glob
import numpy as np
import argparse

# 用于保存gps信息
from PIL import Image
import piexif



from src.API_1GetGpsFromIMG import *
import argparse



# gnss 画图和数据统计
from ex4_gnss_info import *


def Api_getGnssFromImg(input_img_path,Save_GPS_txt):

    # 1读取数据
    Gnss_list=API_read_directory(input_img_path)

    # 2保存txt 格式: name lat lon h
    API_Save2txt(Save_GPS_txt,Gnss_list)

    # 3读取txt 格式: name lat lon h
    #Gnss_list_Read = API_read2txt(GPS_txt_name)



# 
def read_camera_params(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        for key in config:
            print(key,config[key])
    return config

def save_scaled_params(config, output_path):
    with open(output_path, 'w') as file:
        yaml.dump(config, file)

def scale_camera_params(config, r):
    # 缩放后 fx fy cx cy 跟着系数 
    config['Camera.fx'] *= r
    config['Camera.fy'] *= r
    config['Camera.cx'] *= r
    config['Camera.cy'] *= r
    config['Camera.cols'] = int(config['Camera.cols'] * r)
    config['Camera.rows'] = int(config['Camera.rows'] * r)
    # 畸变系数不变，但是因为保存图像是畸变校正后的，所以这里对应的畸变系数是0
    config['Camera.k1'] = 0
    config['Camera.k1'] = 0
    config['Camera.k2'] = 0
    config['Camera.k3'] = 0
    config['Camera.k4'] = 0
    config['Camera.p1'] = 0
    config['Camera.p2'] = 0

    
    return config



def undistort_and_resize_images(input_dir, output_dir, config, r):
    os.makedirs(output_dir, exist_ok=True)
    fx, fy, cx, cy = config['Camera.fx'], config['Camera.fy'], config['Camera.cx'], config['Camera.cy']
    k1, k2, p1, p2 = config['Camera.k1'], config['Camera.k2'], config['Camera.p1'], config['Camera.p2']
    

    cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    #dist_coeffs = np.array([k1, k2, p1, p2])
    dist_coeffs = np.array([0, 0, 0, 0])
    print("==============cam_matrix \n",cam_matrix)
    print("==============dist_coeffs \n",dist_coeffs)
    
    for img_path in glob.glob(os.path.join(input_dir, "*.JPG")):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        new_camera_matrix, roi= cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, cam_matrix, dist_coeffs, None, new_camera_matrix)
        # # 根据ROI进行裁剪，去除无用区域
        # x, y, w, h = roi
        # undistorted_img = undistorted_img[y:y+h, x:x+w]
         
        resized_w = int(w * r)
        resized_h = int(h * r)
        print(resized_w,resized_h)

        resized_img = cv2.resize(undistorted_img, (resized_w, resized_h))
        #resized_img = undistorted_img
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        #cv2.imwrite(output_path, resized_img)
        cv2.namedWindow('Result Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Result Image', resized_img)
        cv2.waitKey(1)

        # 打开图片和读取EXIF数据
        image = Image.open(img_path)
        try:
            exif_dict = piexif.load(img_path)
            exif_bytes = piexif.dump(exif_dict)
        except Exception as e:
            print(f"文件 {img_path} 无EXIF数据或解析失败: {e}")
            exif_bytes = None
            
        # 将BGR格式转换为RGB格式
        pil_rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        # 使用Pillow将其转换为PIL Image对象
        pil_image = Image.fromarray(pil_rgb_image)


        # 保存图片，附加EXIF数据（如果存在）
        if exif_bytes:
            pil_image.save(output_path, exif=exif_bytes,quality=100,optimize=True)
        else:
            pil_image.save(output_path,quality=100)

        print("保存图像",output_path,resized_w,resized_h)

    cv2.destroyAllWindows()



def DO_ONE_DIR(data_path):


    #====================基本不用改路径====================================
    input_camconfig_path = data_path+"/slam_config/GNSS_config_distort.yaml"# 带畸变的原始图像相机内参
    output_camconfig_path =  data_path+"/slam_config/GNSS_config.yaml"# 畸变矫正和缩放后的图像相机内参
    input_images_dir =  data_path+ "/source_images/"# 原始图像(自带gnss信息)文件夹路径
    output_images_dir =  data_path+ "/images/"# 畸变矫正和缩放后的图像文件夹路径
    scaling_ratio = 0.33  # 缩放比例

    #1 提取gnss
    output_gnss_path = data_path+"/slam_config/gnss.txt" # 保存的gnss信息路径
    Api_getGnssFromImg(input_images_dir,output_gnss_path)

    #2 绘制地图
    input_gnsstxt_path = [] # 只画一个轨迹线
    input_gnsstxt_path.append(output_gnss_path)
    output_showgnss_path = data_path+"/slam_config/" #  gps_trajectory_map.html"
    plot_gps_trajectories_on_map(input_gnsstxt_path,output_showgnss_path)

    # 读取并处理相机参数
    camera_config = read_camera_params(input_camconfig_path)
    # 保存缩放后的相机内参
    scaled_camera_config = scale_camera_params(camera_config, scaling_ratio)
    camera_config["Config_PATH"]=data_path
    camera_config["WorkMode"]=WorkMode
    save_scaled_params(scaled_camera_config, output_camconfig_path)

    # 去畸变并保存缩放后的图像
    # undistort_and_resize_images(input_images_dir, output_images_dir, camera_config, scaling_ratio)
    


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

if __name__ == "__main__":

    '''
    WorkMode: map_onlynewData
        # WorkMode_1: map_all_keep_allKfms  建图数据 全体数据参与优化 关键帧不删除全部保留  === 建图用这个
        # WorkMode_2: map_all         建图数据 全体数据参与优化 关键帧自动90%重复删除 == 测试（不如1好）一般不用
        # WorkMode_3: location_only   定位数据 纯定位 不开启建图  == 测试（渲染插帧用不到）一般不用
        # WorkMode_4: map_onlynewData 定位数据 开启建图但是只有新数据参与优化  == 渲染插帧定位用这个
    '''
    WorkMode="map_onlynewData"

    # 文件路径和参数设置
    data_source_path = "/home/dongdong/2project/0data/RTK/data_1_nwpuUp/"
    data_paths = []
    data_paths.append(data_source_path+"/300_260_280/")# 单独需修改一个
    data_paths.append(data_source_path+"/400_500_gps/")# 单独需修改一个
    data_paths.append(data_source_path+"/500_450_gps/")# 单独需修改一个


    
    for path_i in data_paths:
        DO_ONE_DIR(path_i)


