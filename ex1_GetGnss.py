
from src.API_1GetGpsFromIMG import *
import argparse

def Api_getGnssFromImg(input_img_path,Save_GPS_txt):

    # 1读取数据
    Gnss_list=API_read_directory(input_img_path)

    # 2保存txt 格式: name lat lon h
    API_Save2txt(Save_GPS_txt,Gnss_list)

    # 3读取txt 格式: name lat lon h
    #Gnss_list_Read = API_read2txt(GPS_txt_name)

##===========================================================
if __name__ == "__main__":


    # 0 -1 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='处理文件路径参数')

    # 添加参数
    parser.add_argument('--input_img_path', type=str, help='images文件夹的路径')
    parser.add_argument('--Save_GPS_txt', type=str, help='提取的gps保存路径')


    # 解析参数
    args = parser.parse_args()

    # 使用参数
    input_img_path = args.input_img_path
    Save_GPS_txt = args.Save_GPS_txt

    # # 参数
    # data_dir="/home/dongdong/2project/0data/RTK/nwpu/2_gps_300_500_250/300_gps/"

    # # 0-1 gps照片路径
    # input_img_path=data_dir+"images/"
    # # 0-2 txt保存的名字
    # Save_GPS_txt=data_dir+"gnss.txt"

    Api_getGnssFromImg(input_img_path,Save_GPS_txt)

