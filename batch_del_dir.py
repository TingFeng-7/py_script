# -*- coding: utf-8 -*-
"""
给定父目录 递归遍历所有文件
删除指定目录

"""
 
import os
import shutil
from loguru import logger
 
target_paths = [r'D:\sa_data\exps\2022.9.30_yolox_1080p_sa_f27\Element_grabbing\grabbing_evaluation'] # 需要检索的一级 路径
dirNames = ['grabbing_statistics_all_apps'] # 指定目录名
 
 
def delDir(sourcePath):
    items = os.listdir(sourcePath)
    for item in items:
        filePath = os.path.join(sourcePath, item)
        if os.path.isfile(filePath): # 如果是文件 跳过
                continue

        elif os.path.isdir(filePath):
            # logger.info(item)
            if item in dirNames:   # if os.path.splitext(filePath)[1] in postfix: # 后缀名判断
                #递归删除目录
                shutil.rmtree(filePath) # delete
                logger.info('删除成功: ' + filePath)
            else:
                delDir(filePath)
        else:
            print('不是目标文件或文件夹 ' + filePath)
 
if __name__ == '__main__':
    for path in target_paths:
        # sourcePath = path
        delDir(path)