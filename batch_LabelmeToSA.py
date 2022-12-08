# -*- coding: utf-8 -*-
"""
递归遍历所有文件
找到我们想要的目录 ， 把底下目录的json文件全部转换
"""
 
import os
import shutil
from loguru import logger
import json
 
target_paths = ['/workspace/tingfengwu/0001_sa_v1/'] # 需要检索的一级 路径
dirNames = ['yolox_json'] # 指定文件名
#evaluation 文件夹下面

def parse_labelme_json(file_path, image_path=""):
    annos = {}
    boxes_text = []
    boxes_icon = []
    boxes_image = []
    boxes_list = []
    with open(file_path, 'r', encoding='utf-8') as f: #读文件名
        ret_dic = json.load(f)
    if image_path=="":
        act_h = ret_dic["imageHeight"]
        act_w = ret_dic["imageWidth"]
    else:
        image = cv2.imread(image_path)
        act_h, act_w = image.shape[0], image.shape[1]
    annos["imgHeight"] = int(act_h)
    annos["imgWidth"] = int(act_w)
    annos["imageName"] = ret_dic["imagePath"]
    #if self.auto_orc_rec:
    # print(rate_h, rate_w)

    for iter in ret_dic["shape"]:
        remove_idx = None
        cls = iter["label"]
        label_info = {}
        if len(iter["points"]) < 2:
            continue
        [xmin, ymin], [xmax, ymax] = iter["points"] #左下 右上
        b = [int(float(xmin)), int(float(ymin)), int(float(xmax)),
                int(float(ymax))]

        if (b[3] - b[1] <= 0) or (b[2] - b[0] <= 0) or (b[3]>act_h) or(b[2]>act_w):
            print("error rect:", b)
            continue


        label_info["location"] = b


        if cls == "text":
            if remove_idx is not None:
                for ibboxes in boxes_text:
                    if remove_element == ibboxes["location"]:
                        boxes_text.remove(ibboxes)
                        break
            boxes_text.append(label_info)

        elif cls == "icon":
            boxes_icon.append(label_info)
            if remove_idx is not None:
                for ibboxes in boxes_icon:
                    if remove_element == ibboxes["location"]:
                        boxes_icon.remove(ibboxes)
                        break

        elif cls == "image":
            if remove_idx is not None:
                for ibboxes in boxes_image:
                    if remove_element == ibboxes["location"]:
                        boxes_image.remove(ibboxes)
                        break
            boxes_image.append(label_info)
        else:
            print("bad label:", file_path)
    annos["icon"] = boxes_icon
    annos["text"] = boxes_text
    annos["image"] = boxes_image #现在是一个大字典
    
    return annos


def batchTransform(sourcePath):
    items = os.listdir(sourcePath)
    for item in items:
        filePath = os.path.join(sourcePath, item)
        if os.path.isdir(filePath): # 是目录
            # logger.info(item)
            if item in dirNames:   # 是我们想要的目录
                ####################
                for path in os.listdir(filePath): #每个json文件开改
                    sa_json = parse_labelme_json(os.path.join(filePath, path))

                    save_folder = filePath +'/../evaluation/yolox_res/'
                    os.makedirs(save_folder, exist_ok=True)#建一下路径
                    json_name = save_folder + path
                    with open(json_name,'w', encoding='utf-8') as fw: 
                        json.dump(sa_json,fw, indent=2)
                        logger.info('{} saved'.format(json_name)) 
                ########################
                #仿照目录递归生成
                logger.info('转换成功: ' + filePath)
            else:
                batchTransform(filePath) #否则递归去改

        elif os.path.isfile(filePath): # 如果是文件 跳过
                continue
        else:
            print('不是目标文件或文件夹 ' + filePath)
 
if __name__ == '__main__':
    for path in target_paths:
        # sourcePath = path
        batchTransform(path)