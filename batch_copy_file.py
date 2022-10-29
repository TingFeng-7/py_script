# -*- coding: utf-8 -*-
"""
递归遍历所有文件
从源路径 复制指定后缀文件 到目标路径
"""
 
import os
import shutil
from loguru import logger

paths = ['/workspace/bohuang/git_res/YOLOX/exps_pro/2022_8_9_base/elem_detect/f27/'] # 需要检索的一级 路径
savePath = '/YOLOX/hbo_exps/' # 目标路径
# 使用方式
postfix = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG'] # 1.指定文件后缀名
fileNames = ['exp.py'] # 2.指定文件名
 
if not os.path.exists(savePath):
    os.mkdir(savePath)
 
def copyFile(sourcePath, savePath):
    items = os.listdir(sourcePath)
    for item in items:
        filePath = os.path.join(sourcePath, item)
        if os.path.isfile(filePath): # 已经是 文件了
            filePath_s = str(filePath)
            filename = filePath_s.split('/')[-1]
            
            if filename in fileNames: 
            # if os.path.splitext(filePath)[1] in postfix: # 后缀名判断
                logger.info(filename)
                #仿照目录递归生成
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                shutil.copyfile(filePath, os.path.join(savePath,item)) # 复制文件到目标文件夹
                logger.info(' 复制成功 ' + filePath + "复制到"+ os.path.join(savePath,item))
            else:
                continue
        elif os.path.isdir(filePath):
            # copyFile(filePath, savePath)
            copyFile(filePath, os.path.join(savePath,item)) # 如果是文件夹，则再次调用此函数，递归处理 每次深入都要再链接名字
        else:
            print('不是目标文件或文件夹 ' + filePath)

def copyFile_pathlib(sourcePath, savePath):
    items = os.listdir(sourcePath)
    for item in items:
        filePath = os.path.join(sourcePath, item)
        if os.path.isfile(filePath): # 已经是 文件了
            filePath_s = str(filePath)
            filename = filePath_s.split('/')[-1]
            
            if filename in fileNames: 
            # if os.path.splitext(filePath)[1] in postfix: # 后缀名判断
                logger.info(filename)
                #仿照目录递归生成
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                shutil.copyfile(filePath, os.path.join(savePath,item)) # 复制文件到目标文件夹
                logger.info(' 复制成功 ' + filePath + "复制到"+ os.path.join(savePath,item))
            else:
                continue
        elif os.path.isdir(filePath):
            # copyFile(filePath, savePath)
            copyFile(filePath, os.path.join(savePath,item)) # 如果是文件夹，则再次调用此函数，递归处理 每次深入都要再链接名字
        else:
            print('不是目标文件或文件夹 ' + filePath)
        return 0

if __name__ == '__main__':
    for path in paths:
        sourcePath = path
        copyFile(sourcePath, savePath)