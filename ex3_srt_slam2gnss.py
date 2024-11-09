import numpy as np
import src.API_2Gps22ENU as API_2Gps22ENU
import src.API_33DTo3D as API_33DTo3D
import src.API_4DrawPic as API_4DrawPic



import cv2
import argparse

# pip install pyyaml
import yaml

import math


def API_Save2txt(txt_name,Gnss_list):

    with open(txt_name, 'w') as file:
        for row in Gnss_list:
            line = ' '.join(map(str, row))
            file.write(f"{line}\n")

    print(txt_name,"保存成功")



def read_slamPose_Text(txt_name):
    '''
    示例字典
    GNSS_LIST = {
        '2024-01-01T12:00:00Z': [34.052235, -118.243683, 100],
        '2024-01-01T12:05:00Z': [36.169941, -115.139832, 200],
        '2024-01-01T12:10:00Z': [37.774929, -122.419418, 300]
    }
    '''
    Slam_ENU_LIST={}

    with open(txt_name, "r") as f1: # 读取所有镇的gps信息，
        
        for line in f1:
            line = line.strip()
            line = line.replace("  ", " ")
            elems = line.split(" ")  # 0 图像名称（不带后缀）1-3 lat lon alt
            name = str(elems[0])
            #init_gnss=[init_lat,init_lon,init_h]

            x=float(elems[1])
            y=float(elems[2])
            z=float(elems[3])
          
            Slam_ENU_LIST[name]=[x,y,z]

    return Slam_ENU_LIST



def API_read_from_txt(filename):
    """
    从指定文件中读取数据并解析为字典列表。

    :param filename: 数据文件的路径
    :return: 包含数据字典的列表，每个字典包含时间戳、纬度、经度和高度
    """
    data = []

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamp = parts[0]  # 时间戳
                latitude = float(parts[1])  # 纬度
                longitude = float(parts[2])  # 经度
                altitude = float(parts[3])  # 高度
                
                data.append({
                    'timestamp': timestamp,
                    'latitude': latitude,
                    'longitude': longitude,
                    'altitude': altitude
                })
                
    except FileNotFoundError:
        print(f"文件 {filename} 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

    return data


def quaternion_conjugate(qvec):
    """计算四元数的共轭"""
    return np.array([qvec[0], -qvec[1], -qvec[2], -qvec[3]])


def quaternion_rotate_vector(qvec, vec):
    """使用四元数旋转一个向量"""
    qvec_w, qvec_x, qvec_y, qvec_z = qvec
    # 将向量表示为四元数 [0, x, y, z]
    vec_quat = np.array([0, vec[0], vec[1], vec[2]])
    # 四元数乘法
    q_conj = quaternion_conjugate(qvec)

    # 四元数相乘的公式 q * v * q^-1
    vec_rotated = quaternion_multiply(
        quaternion_multiply(qvec, vec_quat), q_conj
    )

    # 返回旋转后的向量
    return vec_rotated[1:]


def quaternion_multiply(q1, q2):
    """四元数相乘"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    ])


def camera_to_world(qvec, tvec):
    """将 qvec 和 tvec 从相机坐标系转换到世界坐标系"""
    # 1. 计算四元数的共轭
    qvec_conj = quaternion_conjugate(qvec)

    # 2. 旋转 tvec 并取反
    tvec_world = -quaternion_rotate_vector(qvec_conj, tvec)

    return qvec_conj, tvec_world

def find_timestamp(filename):

    dot_index = filename.rfind('.')
    image_name = filename[0:dot_index]
    
    return image_name

def read_colmapPose_imageText(path):
    """
    参考自 https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    dict_video_colmap_xyz = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#" \
               and (line.find("jpg")!=-1 or line.find("png")!=-1 or line.find("JPG")!=-1):
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]

                # colmap默认的是从世界到相机，此处转换 qvec 和 tvec 从相机到世界
                qvec_wc, tvec_wc = camera_to_world(qvec, tvec)

                #time_stamp = find_timestamp(image_name)

                # 保存相机在colmap世界坐标系下的位置到文件txt，格式为time_stamp x y z
                dict_video_colmap_xyz[image_name] = [tvec_wc[0],tvec_wc[1],tvec_wc[2]]

    return dict_video_colmap_xyz


def Read_gnss_ref_from_yaml(gps_config_yaml):
    init_lat=0
    init_lon=0
    init_h=0

    with open(gps_config_yaml, 'r') as file:
        data = yaml.safe_load(file)
        if isinstance(data, dict):
            # 提取数据
            init_lat = data.get('Initial.lat', None)
            init_lon = data.get('Initial.lon', None)
            init_h = data.get('Initial.alt', None)
        else:
            print("读取GNSS原始点失败")
    
    print("初始参考点",init_lat,' ',init_lon,' ',init_h)
    init_gnss=[init_lat,init_lon,init_h]
    return init_gnss






def read_gnssPose_Text(txt_name,init_gnss):
    '''
    示例字典
    GNSS_LIST = {
        '2024-01-01T12:00:00Z': [34.052235, -118.243683, 100],
        '2024-01-01T12:05:00Z': [36.169941, -115.139832, 200],
        '2024-01-01T12:10:00Z': [37.774929, -122.419418, 300]
    }
    '''
    GNSS_ENU_LIST={}
    GNSS_LIST={}
    with open(txt_name, "r") as f1: # 读取所有镇的gps信息，
        
        for line in f1:
            line = line.strip()
            line = line.replace("  ", " ")
            elems = line.split(" ")  # 0 图像名称（不带后缀）1-3 lat lon alt
            time_stamp = str(elems[0])
            #init_gnss=[init_lat,init_lon,init_h]

            lat=float(elems[1])
            lon=float(elems[2])
            alt=float(elems[3])
            gnss_in=[lat,lon,alt]
            GNSS_LIST[time_stamp]=gnss_in

            e, n, u = API_2Gps22ENU.API_gnss_to_enu(gnss_in,init_gnss)

            

       
            GNSS_ENU_LIST[time_stamp] = [e,n,u]
            

    return GNSS_ENU_LIST,GNSS_LIST



if __name__ == "__main__":
    # 0 -1 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='处理文件路径参数')

    # 添加参数
    parser.add_argument('--slam_txt', type=str, help='images.txt文件的路径')
    parser.add_argument('--gps_intPionts', type=str, help='gps NED坐标系的原点xml文件的路径')
    parser.add_argument('--gps_txt', type=str, help='视频帧的gps信息文件的路径')
    parser.add_argument('--slam2gnss_SRt_xml', type=str, help='输出的xml文件路径')

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    slam_txt = args.slam_txt
    gps_intPionts = args.gps_intPionts
    gps_txt = args.gps_txt
    slam2gnss_SRt_xml = args.slam2gnss_SRt_xml

    # ========= 1 - 1 解析 colamp 位姿结果  通过colmap结果查找对应的GPS 
    #WCOLMAP_ENU_LIST = read_colmapPose_imageText(slam_txt)

    Slam_ENU_LIST = read_slamPose_Text(slam_txt)
    
    

    # ========= 1-2-1 读入gnss ned坐标系原点
    # 读取 YAML 文件
    init_gnss=Read_gnss_ref_from_yaml(gps_intPionts)

    # ========= 1-2-2 解析 gnss 位姿结果
    GNSS_ENU_LIST,GNSS_LIST = read_gnssPose_Text(gps_txt,init_gnss)

    # ========= 2-1 按时间戳(str)匹配GNSS和colmap数据
    points_src_colmap=[]
    points_dst_gnss=[]
    points_id_timeshap=[]
    for gnss_name in GNSS_ENU_LIST.keys():
        #print("gnss-enu",GNSS_ENU_LIST[gnss_name],"",dict_video_colmap_xyz[gnss_name])
        if Slam_ENU_LIST.get(gnss_name):
            points_src_colmap.append(Slam_ENU_LIST[gnss_name])
            points_dst_gnss.append(GNSS_ENU_LIST[gnss_name])
            points_id_timeshap.append(gnss_name)
            print(gnss_name)
    gps_ned_np = np.array(points_src_colmap)
    colmap_ned_np = np.array(points_dst_gnss)

    #gps_ned_np = np.array(points_dst_gnss)
    #colmap_ned_np = np.array(points_src_colmap)


    # ========= 2-2 计算变换关系
    s, R, sR,t  = API_33DTo3D.API_pose_estimation_3dTo3d_ransac(points_src_colmap, points_dst_gnss) # 
    #
    #s, R, sR,t  = API_33DTo3D.umeyama_alignment(gps_ned_np.T, colmap_ned_np.T) # EVO自带的 
   


    #========= 3-1 将 R T s分别写入xml中
    fs = cv2.FileStorage(slam2gnss_SRt_xml, cv2.FILE_STORAGE_WRITE)
    fs.writeComment('这里的sRT是从colmap到GPS-NED坐标系下的', 0)
    fs.write('R', R)
    fs.write('t', t)
    fs.write('s', s)
    fs.release()


    # ======== 3-2 从 YAML 文件中读取数据
    fs = cv2.FileStorage(slam2gnss_SRt_xml, cv2.FILE_STORAGE_READ)
    # 读取旋转矩阵 R
    R = fs.getNode('R').mat()


    # 计算行列式
    #det_R = np.linalg.det(R)

    # 读取平移向量 t
    t = fs.getNode('t').mat()
    # 读取尺度 s
    s = fs.getNode('s').real()
    # 释放文件存储
    fs.release()

    # 打印读取的数据以验证
    print('旋转矩阵 R:')
    print(R)
    print('位移向量 t:')
    print(t)
    print('尺度因子 s:')
    print(s)
    # 生成尺度矩阵
    S = np.diag([s, s, s])

    # 组合旋转矩阵、尺度矩阵和位移向量成变换矩阵
    # 变换矩阵的形式为 [R*s | t]
    # 其中 R*s 为旋转缩放矩阵，t 为平移向量
    T = np.hstack((S @ R, t.reshape(-1, 1)))

    # 将 T 变换矩阵扩展为 4x4 矩阵
    T_homogeneous = np.vstack((T, [0, 0, 0, 1]))

    # 打印变换矩阵
    print('变换矩阵 T:')
    print(T_homogeneous)


    #===========  4 保存计算结果 colmap enu变换到gnss enu坐标系下的新坐标
    colmapenu_in_gnssenu_xyz = API_33DTo3D.API_src3D_sRt_dis3D_list(points_src_colmap, points_dst_gnss, sR, t)
    
    draw_tracelist=[]
    draw_tracelist.append(colmapenu_in_gnssenu_xyz)
    draw_tracelist.append(points_dst_gnss)
    API_4DrawPic.Draw3D_trace_more(draw_tracelist)

    gnss_enu =[]
    colmapenu_in_gnssenu_tenu=[]
    for i in range(0,len(points_id_timeshap)):
        timeshap= points_id_timeshap[i]
        #保存数据 名字 e n u
        li=[timeshap,colmapenu_in_gnssenu_xyz[i][0],colmapenu_in_gnssenu_xyz[i][1],colmapenu_in_gnssenu_xyz[i][2]]
        colmapenu_in_gnssenu_tenu.append(li)

        gi=[timeshap,points_dst_gnss[i][0],points_dst_gnss[i][1],points_dst_gnss[i][2]]
        gnss_enu.append(gi)

    # 保存数据

    #colmape_selfEnu_txt_name
    colmape_GnssEnu_txt_name="11_colmape_GnssEnu.txt"
    #API_Save2txt(colmape_GnssEnu_txt_name,colmapenu_in_gnssenu_tenu)    
    
    Gnss_enu_txt_name="0_gnss_enu.txt"
    #API_Save2txt(Gnss_enu_txt_name,gnss_enu)



    # 4 读取enu数据 转化到 gnss
    # 4-1 获取gnss参考点 - 名字 纬 经 高

    gnss_ref=init_gnss

    print("参考GNSS位置",gnss_ref)
    # 4-2 获取enu数据集 -名字 e n u
    #enu_list_Read=API_read2txt(colmape_GnssEnu_txt_name)
    enu_list_Read=colmapenu_in_gnssenu_tenu
    # 4-3 ENU数据转化为gnss数据
    GNSS_list_from_enu=API_2Gps22ENU.API_enu4_to_gnss4_list(enu_list_Read,gnss_ref)
    # 4-2 保存gnss结果 名字 纬 经 高
    colmap_Gnss_txt_name="12_colmap_Gnss.txt"
    #API_Save2txt(colmap_Gnss_txt_name,GNSS_list_from_enu)