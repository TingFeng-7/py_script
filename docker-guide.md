## 看看dockefile

## 1. -f 以dockerfile 构建镜像
docker build -f Dockerfile -t yolox-detector:v1.0 .
## 2. docker pull 直接拉


## run container
 
- docker

- docker 19 以前：
  nvidia-docker run -it --name yolox-detector  yolox-detector:v1.0 /bin/bash
参数必须写在前面才动起来
  nvidia-docker run -v /workspace:/workspace -v /data:/data -it  --name yolox-detector2  yolox-detector:v2.0  /bin/bash
 ## docker版本问题

 在docker19以前都需要单独下载nvidia-docker1或nvidia-docker2来启动容器，自从升级了docker19后跑需要gpu的docker只需要加个参数–gpus all 即可 为了让docker支持nvidia显 …
 nvidia-docker 启动

 ## 退出容器 记住id
    da770501237f
vscode extension: remote container + docker

 ## 重新进入容器
    docker container exec -it da770501237f /bin/bash
    root@3027eb644874:/#

## 将容器保存为镜像

docker commit da770501237f yolox-detector:v2.0

## run yolox demo
- python tools/demo.py image -n yolox-s -c /path/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]

` python tools/demo.py image -f experiments/1088exp.py -c experiments/epoch_300_ckpt.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu `

python tools/demo.py 
- image 
- -f (network config)  boge_experiments/f32/1080_no_enhance/exp.py 
- -c(model) boge_experiments/f32/1080_no_enhance/latest_ckpt.pth 
- --path(imgs directory) /workspace/tingfengwu/0001_sa_v1/0001_cyclone_1.5/imgs/
- --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu 

## docker 挂载目录 -v


## yolox 模型实验文件
/workspace/bohuang/git_res/YOLOX/exps_pro
## 建立软连接
ln –s  /workspace/bohuang/git_res/YOLOX/exps_pro  bohuang_exps

## exp.py
/workspace/bohuang/git_res/YOLOX/exps_pro/2022_8_9_base/elem_detect/f32/1080_no_enhance/exp.py
##
/workspace/bohuang/git_res/YOLOX/exps_pro/2022_8_9_base/elem_detect/f32/1080_no_enhance/YOLOX_outputs/exp/latest_epoch_ckpt.pth


## 批量运行 测试全部27个数据集
python tools/demo.py image_all -f hbo_exps/gt1_fpn4_lr0.01_1088/exp.py  -c /workspace/bohuang/git_res/YOLOX/exps_pro/2022_8_9_base/elem_detect/f27/gt1_fpn4_lr0.01/YOLOX_outputs/exp/latest_ckpt.pth --path /workspace/tingfengwu/0001_sa_v1/ --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu