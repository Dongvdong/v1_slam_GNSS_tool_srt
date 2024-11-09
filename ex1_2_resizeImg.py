import cv2
import yaml
import os
import glob
import numpy as np
import argparse

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
    dist_coeffs = np.array([k1, k2, p1, p2])
    print("cam_matrix \n",cam_matrix)
    print("dist_coeffs \n",dist_coeffs)
    
    for img_path in glob.glob(os.path.join(input_dir, "*.JPG")):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        new_camera_matrix, roi= cv2.getOptimalNewCameraMatrix(cam_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, cam_matrix, dist_coeffs, None, new_camera_matrix)
        # 根据ROI进行裁剪，去除无用区域
        x, y, w, h = roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]
         
        resized_w = int(w * r)
        resized_h = int(h * r)

        resized_img = cv2.resize(undistorted_img, (resized_w, resized_h))
        #resized_img = undistorted_img
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, resized_img)
        print("保存图像",output_path,resized_w,resized_h)
        cv2.namedWindow('Result Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Result Image', resized_img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":


    # # 0 -1 创建 ArgumentParser 对象
    # parser = argparse.ArgumentParser(description='处理文件路径参数')

    # # 添加参数
    # parser.add_argument('--input_camConfig_path', type=str, help='输入相机内参文件')
    # parser.add_argument('--output_camConfig_path', type=str, help='输处畸变矫正和缩放后的相机内参文件')
    # parser.add_argument('--input_images_dir', type=str, help='原始图像文件夹路径')
    # parser.add_argument('--output_images_dir',type=str, help='畸变矫正和缩放后的图像文件夹路径')
    # parser.add_argument('--scaling_ratio', type=str, help='缩放倍数')
    # # 解析参数
    # args = parser.parse_args()

    # input_camconfig_path = args.input_camconfig_path
    # output_camconfig_path = args.o2utput_camconfig_path
    # input_images_dir = args.input_images_dir
    # output_images_dir = args.output_images_dir
    # scaling_ratio = args.scaling_ratio

    # 文件路径和参数设置
    data_path="/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/500_450_gps"
    input_camconfig_path = data_path+"/原始数据/GNSS_config_distort.yaml"
    output_camconfig_path =  data_path+"/GNSS_config.yaml"
    input_images_dir =  data_path+ "/原始数据/images"
    output_images_dir =  data_path+ "/images"
    scaling_ratio = 0.33  # 示例比例

    # 读取并处理相机参数
    camera_config = read_camera_params(input_camconfig_path)
   
    # 去畸变并保存缩放后的图像
    undistort_and_resize_images(input_images_dir, output_images_dir, camera_config, scaling_ratio)
    # 保存缩放后的相机内参
    scaled_camera_config = scale_camera_params(camera_config, scaling_ratio)
    save_scaled_params(scaled_camera_config, output_camconfig_path)