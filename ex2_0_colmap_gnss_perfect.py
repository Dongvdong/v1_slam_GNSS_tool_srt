import os
import subprocess
import numpy as np
from sqlite3 import connect

# 设置路径
project_dir = "/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/300_gps/colamp_gnss/"  # COLMAP 项目目录
image_dir = project_dir+"/images/"    # 图像目录
database_path = os.path.join(project_dir, "database.db")  # COLMAP 数据库路径
output_dir = os.path.join(project_dir, "sparse")  # 输出目录 后面自动加 0 文件夹


gnss_data_path = project_dir + "/gnss.txt"  # GNSS 数据文件路径

# GPU 加速设置
gpu_index = 0  # 默认使用第一个GPU，若有多个GPU可修改该值

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 2. 创建COLMAP数据库
if os.path.exists(database_path):
    subprocess.run(["rm", "-rf", database_path])  # 删除现有数据库

subprocess.run([
    "colmap", "database_creator", "--database_path", database_path
])


# 1. 从 TXT 文件读取 GNSS 数据
def read_gnss_data(gnss_data_path):
    gnss_data = {}
    with open(gnss_data_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 4:
                image_name = parts[0].strip()
                latitude = float(parts[1].strip())
                longitude = float(parts[2].strip())
                altitude = float(parts[3].strip())
                gnss_data[image_name] = (latitude, longitude, altitude)
                print("读取到GNSS", image_name, latitude, longitude, altitude)
    return gnss_data

#2 写入GNSS信息
def write_gnss_data(database_path, image_dir,gnss_data):
    for image_name, (latitude, longitude, altitude) in gnss_data.items():
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            subprocess.run([
                "colmap", "gps_importer",
                "--database_path", database_path,
                "--image_path", image_path,
                "--latitude", str(latitude),
                "--longitude", str(longitude),
                "--altitude", str(altitude)
            ])





# 2. 提取图像特征（启用 GPU 加速）
def extract_features(database_path, image_dir, gpu_index):
    print("Extracting features with GPU acceleration...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "true",
        "--ImageReader.camera_model","PINHOLE",
        "--SiftExtraction.use_gpu", "true"  # 开启 GPU 加速
    ])

# 3. 逐对匹配图像（按顺序）
def match_images(database_path, image_dir):
    print("Matching images in sequence...")
    
    # # 获取所有图像文件的名称并按字母顺序排列
    # image_files = sorted(os.listdir(image_dir))  # 按字母顺序排列图像

    # # 遍历每一对图像进行匹配
    # for i in range(len(image_files)):
    #     for j in range(i + 1, len(image_files)):
    #         image1 = image_files[i]
    #         image2 = image_files[j]
    #         print(f"Matching {image1} with {image2}...")
    # 自动顺序匹配
    subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", database_path, 
        "--SiftMatching.guided_matching","true",
        "--SiftMatching.use_gpu", "true"  # 开启 GPU 加速
    ])

# 4. 运行 COLMAP 命令：稀疏重建（启用 GPU 加速）
def sparse_reconstruction(database_path, image_dir, output_dir, gpu_index):
    print("Running sparse reconstruction with GPU acceleration...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,  # 注意：这里指定了图像目录路径
        "--output_path", output_dir, # 后面自动追加0
       
       
    ])

# 5. 运行 COLMAP 命令：束束调整（优化，启用 GPU 加速）
def bundle_adjustment(output_dir, gpu_index):
    print("Running bundle adjustment with GPU acceleration...")
    subprocess.run([
        "colmap", "bundle_adjuster",
        "--input_path", output_dir+"/0/",
        "--output_path", output_dir+"/0/",
       
      
    ])



# 6. 更新数据库并加入 GNSS 数据作为约束
def update_database_with_gnss(database_path, gnss_data):
    print("Updating database with GNSS data...")

    conn = connect(database_path)
    cursor = conn.cursor()

    # 遍历所有图像，将 GNSS 数据作为相机外参约束
    for image_name, (latitude, longitude, altitude) in gnss_data.items():
        cursor.execute("""
            SELECT image_id FROM images WHERE name = ?
        """, (image_name,))
        image_id = cursor.fetchone()
        
        if image_id:
            # GNSS 数据转化为相机位姿
            #camera_position = (latitude, longitude, altitude)  # 示例：直接使用经纬度和高度
            
            # 更新相机的位置信息（这里只是一个简化示例，实际中可能涉及坐标系转换）
            # cursor.execute("""
            #     UPDATE images
            #     SET prior_qw = ?, prior_qx = ?, prior_qy = ?, prior_qz = ?, prior_tx = ?, prior_ty = ?, prior_tz = ?
            #     WHERE image_id = ?
            # """, (0, 0, 0, 1, *camera_position, image_id[0]))  # 假设四元数是单位四元数
             # 更新图像的 GPS 数据
            cursor.execute("""
                UPDATE images
                SET latitude = ?, longitude = ?, altitude = ?
                WHERE name = ?
            """, (latitude, longitude, altitude, image_name))
            print("加入gnss数据", image_name,latitude, longitude, altitude)
        else:
            print(f"No image found with name {image_name}")


    conn.commit()
    conn.close()

# 7. 导出结果为 TXT 格式（分为三部分）
def export_results(output_dir):
    print("Exporting results to separate TXT files...")

    # 导出为相机信息
   
    subprocess.run([
        "colmap", "model_converter",
        "--input_path", output_dir+"/0/",
        "--output_path", output_dir+"/0/",
        "--output_type", "txt"
    ])

   

# 8. 检查重建结果（可选）
def visualize_results(project_dir):

    print("Visualizing results...",project_dir)
    subprocess.run(["colmap", "gui", 
                    "--project_path", project_dir,
                    "--database_path", project_dir+"/database.db",
                    
                    ])

# 主流程
def run_colmap_process():
    # 1. 读取 GNSS 数据
    gnss_data = read_gnss_data(gnss_data_path)

    # 4. 更新数据库并加入 GNSS 数据作为约束
    write_gnss_data(database_path,image_dir,gnss_data)
    #update_database_with_gnss(database_path, gnss_data)


    # 2. 提取特征
    extract_features(database_path, image_dir, gpu_index)

    # 3. 顺序匹配图像
    match_images(database_path, image_dir)

    # 5. 稀疏重建
    sparse_reconstruction(database_path, image_dir, output_dir, gpu_index)

    # 6. 优化（束束调整）
    bundle_adjustment(output_dir, gpu_index)

    # 7. 导出结果
    export_results(output_dir)

    # 8. 可选：检查重建结果
    # visualize_results(output_dir)

if __name__ == "__main__":
    run_colmap_process()
