import pandas as pd
import folium

def read_gps_data_from_txt(file_path):
    """
    读取 GPS 数据文件，将数据解析为 (filename, latitude, longitude, altitude) 格式的列表。
    :param file_path: .txt 文件路径
    :return: 包含 GPS 数据的 DataFrame
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                filename, lat, lon, alt = parts
                data.append((filename, float(lat), float(lon), float(alt)))
    return pd.DataFrame(data, columns=['filename', 'latitude', 'longitude', 'altitude'])

def plot_gps_trajectories_on_map(input_gnsstxt_path,output_showgnss_path):




    """
    从多个 txt 文件中读取 GPS 数据并在同一张地图上绘制轨迹，每条轨迹使用不同颜色。
    :param file_paths: 一个或多个 txt 文件的路径
    :return: 生成并保存的地图 HTML 文件
    """
    # 创建地图中心点，选择第一个文件的第一个数据点作为中心
    df1 = read_gps_data_from_txt(input_gnsstxt_path[0])
    map_center = [df1['latitude'].iloc[0], df1['longitude'].iloc[0]]
    mymap = folium.Map(location=map_center, zoom_start=16)

    # 定义颜色列表 blue
    colors = ['green', 'blue', 'red', 'purple', 'purple',"purple",'purple', 'purple',"orange"]#orange

    # 绘制每个轨迹到地图
    for i, file_path in enumerate(input_gnsstxt_path):
        df = read_gps_data_from_txt(file_path)
        color = colors[i % len(colors)]  # 防止轨迹超过颜色列表长度
        folium.PolyLine(locations=df[['latitude', 'longitude']].values, color=color, weight=3, opacity=0.7).add_to(mymap)


        results = process_gps_data(file_path,output_showgnss_path)


    # 保存地图为 HTML 文件
    #map_filename = "gps_trajectory_map.html"
    map_filename=output_showgnss_path+"gps_trajectory_map.html"
    mymap.save(map_filename)
    print(f"地图已保存为 {map_filename}")
    return mymap



import math

# Haversine公式计算两点之间的地面距离
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径，单位：公里
    # 将经纬度从度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # 返回距离，单位：公里

# 读取数据并解析
def process_gps_data(input_gnss_file_path,out_info_path):
    with open(input_gnss_file_path, 'r') as file:
        lines = file.readlines()

    gps_data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            filename, lat, lon, alt = parts
            gps_data.append((filename, float(lat), float(lon), float(alt)))

    # 数据数量
    data_count = len(gps_data)
    
    # 计算总里程
    total_distance = 0
    for i in range(1, len(gps_data)):
        lat1, lon1, alt1 = gps_data[i-1][1], gps_data[i-1][2], gps_data[i-1][3]
        lat2, lon2, alt2 = gps_data[i][1], gps_data[i][2], gps_data[i][3]
        total_distance += haversine(lat1, lon1, lat2, lon2)
    
    # 计算平面区域长宽
    latitudes = [point[1] for point in gps_data]
    longitudes = [point[2] for point in gps_data]
    
    min_lat = min(latitudes)
    max_lat = max(latitudes)
    min_lon = min(longitudes)
    max_lon = max(longitudes)
    
    # 纬度每度大约 111.32 公里
    lat_length = (max_lat - min_lat) * 111320  # 长度（米）
    
    # 经度每度距离 = 111320 * cos(纬度)
    avg_lat = (max_lat + min_lat) / 2
    lon_length = (max_lon - min_lon) * 111320 * math.cos(math.radians(avg_lat))  # 宽度（米）
    

    # 计算平面覆盖范围（面积）
    plane_area = lat_length * lon_length  # 面积（平方米）

    # 计算高度范围
    altitudes = [point[3] for point in gps_data]
    min_alt = min(altitudes)
    max_alt = max(altitudes)
    height_range = max_alt - min_alt  # 高度差（米）
    
    # 计算空域体积
    area = lat_length * lon_length  # 面积（平方米）
    volume = area * height_range  # 体积（立方米）

    
    results ={
        'data_count': data_count,
        'total_distance_km': total_distance,
        'lat_length_m': lat_length,
        'lon_length_m': lon_length,
        'plane_area_m2': plane_area,  # 平面覆盖范围
        'min_alt_m': min_alt,  # 最小高度
        'max_alt_m': max_alt,  # 最大高度
        'height_range_m': height_range,
        'airspace_volume_m3': volume
    }

    # 保存结果到文件
    output_file = out_info_path+'/场景大小信息.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"数据数量: {results['data_count']}\n")
        f.write(f"总里程长度: {results['total_distance_km']:.2f} 公里\n")
        f.write(f"平面区域长度: {results['lat_length_m']:.2f} 米\n")
        f.write(f"平面区域宽度: {results['lon_length_m']:.2f} 米\n")
        f.write(f"平面覆盖范围: {results['plane_area_m2']:.2f} 平方米\n")
        f.write(f"最小海拔高度: {results['min_alt_m']:.2f} 米\n")
        f.write(f"最大海拔高度: {results['max_alt_m']:.2f} 米\n")
        f.write(f"参考点海拔高度: {423} 米\n")
        f.write(f"高度范围: {results['height_range_m']:.2f} 米\n")
        f.write(f"空域体积: {results['airspace_volume_m3']:.2f} 立方米\n")
    

    # 打印结果
    print(f"数据数量: {results['data_count']}")
    print(f"总里程长度: {results['total_distance_km']:.2f} 公里")
    print(f"平面区域长度: {results['lat_length_m']:.2f} 米")
    print(f"平面区域宽度: {results['lon_length_m']:.2f} 米")
    print(f"平面覆盖范围: {results['plane_area_m2']:.2f} 平方米")
    print(f"最小海拔高度: {results['min_alt_m']:.2f} 米")
    print(f"最大海拔高度: {results['max_alt_m']:.2f} 米")
    print(f"参考点海拔高度: {423} 米")  # 假设参考点海拔高度为 423 米
    print(f"高度范围: {results['height_range_m']:.2f} 米")
    print(f"空域体积: {results['airspace_volume_m3']:.2f} 立方米")

    return results



# if __name__ == "__main__":


#     dir_path = '/home/dongdong/2project/0data/RTK/data2_nwpuFly/'

#     # 示例：调用上述函数来读取数据并绘制轨迹


#     file_paths=[]


#     file_paths.append(dir_path+"300_建图1_2pm/gnss.txt")    
#     #file_paths.append(dir_path+"300_建图2/gnss.txt")
#     #file_paths.append(dir_path+"300_map3_5pm/gnss.txt")

#     file_paths.append(dir_path+"300_COLMAP_map/gnss.txt")

#     #file_paths.append(dir_path+"260_280/gnss.txt")
#     #file_paths.append(dir_path+"300-280-260/gnss.txt")
#     # file_paths.append(dir_path+"320-340-360/gnss.txt")
#     # file_paths.append(dir_path+"400-440/gnss.txt")
#     # file_paths.append(dir_path+"500/gnss.txt")


#     # 绘制地图
#     plot_gps_trajectories_on_map(file_paths)
    
