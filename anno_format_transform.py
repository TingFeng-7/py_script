from paddleocr import PaddleOCR
import os
import sys
import json
import argparse
import cv2
import tqdm
import base64
import onnxruntime
import shutil

classes = ["text", "icon", "image"]

key_value_dict = dict()

def remove_dir(dst_path):
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)

def sa_json2std(src_path, dst_path, prefix_relative_path, std_split=False):
    imgs_name = os.path.split(src_path)[-1].split(".")[0]
    if std_split:
        image_save_path = prefix_relative_path+'/'+imgs_name+'/img.png'
    else:
        image_save_path = prefix_relative_path+"/imgs/"+imgs_name+'.png'

    anno = {}
    with open(src_path, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
        # image_save_path = ret_dic["imageName"]
        total_anno = []
        icon = ret_dic["icon"]
        text = ret_dic["text"]
        image = ret_dic["image"]
        for iter in icon:
            total_anno.append(["icon", int(iter["location"][0]), int(iter["location"][1]), int(iter["location"][2]),
                               int(iter["location"][3])])
        for iter in text:
            total_anno.append(["text", int(iter["location"][0]), int(iter["location"][1]), int(iter["location"][2]),
                               int(iter["location"][3])])
        for iter in image:
            total_anno.append(
                ["image", int(iter["location"][0]), int(iter["location"][1]), int(iter["location"][2]),
                 int(iter["location"][3])])

        with open(dst_path, "w", encoding='utf-8') as label_file:
            label_file.write(image_save_path + '\n')
            for label, left, top, right, bottom in total_anno:
                msg = "\t".join(
                    [label, str(left), str(top), str(right), str(bottom)])
                label_file.write(msg + '\n')


def walk_folder_search_file(root_path, file_type):
    file_path_list = []
    for root, dirs, files in os.walk(root_path):
        for f in files:
            if f.endswith(file_type):
                file_path_list.append(os.path.join(root, f))
    
    return file_path_list


def cal_area(box):
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    area = w * h
    return area

def crop_icon_text(sa_json_path, image_path, subimg_dir, ocr_tool, padding_size=3):
    # print(sa_json_path)
    time_stamp_name = os.path.split(sa_json_path)[1].split(".json")[0]
    with open(sa_json_path, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
        # image_name = ret_dic["imageName"]
        icon = ret_dic["icon"]
        text = ret_dic["text"]
        imgs = ret_dic["image"]
        image = cv2.imread(image_path)
        if(len(icon) > 0):
            icon_image_path = os.path.join(subimg_dir,"icon") #'%s/icon/' % (subimg_dir)
            if not os.path.exists(icon_image_path):
                os.makedirs(icon_image_path)
            for iter in icon:
                b = [int(iter["location"][0]), int(iter["location"][1]), int(
                    iter["location"][2]), int(iter["location"][3])]
                padding_b = [max(b[0] - padding_size, 0), max(b[1] - padding_size, 0), min(
                    b[2] + padding_size, image.shape[1] - 1), min(b[3] + padding_size, image.shape[0] - 1)]
                roi_img = image[padding_b[1]: padding_b[3],
                                padding_b[0]:padding_b[2]]
                image_path = '%s/%s@%sx%sx%sx%s.png' % (
                    icon_image_path, time_stamp_name, int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                cv2.imwrite(image_path, roi_img)
        if(len(text) > 0):
            text_image_path = os.path.join(subimg_dir,"text")  #'%s/text/' % (subimg_dir)
            if not os.path.exists(text_image_path):
                os.makedirs(text_image_path)
            unrec_text_path = '%s/unrec/' % (text_image_path)
            if not os.path.exists(unrec_text_path):
                os.makedirs(unrec_text_path)
            for iter in text:
                b = [int(iter["location"][0]), int(iter["location"][1]), int(iter["location"][2]),
                     int(iter["location"][3])]
                padding_b = [max(b[0] - padding_size, 0), max(b[1] - padding_size, 0),
                             min(b[2] + padding_size, image.shape[1] - 1), min(b[3] + padding_size, image.shape[0] - 1)]
                roi_img = image[padding_b[1]: padding_b[3],
                                padding_b[0]:padding_b[2]]
                if len(iter["content"]) > 0:
                    save_path = '%s/%s@%sx%sx%sx%s.png' % (
                        text_image_path, time_stamp_name, int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    # save ocr label
                    ocr_json = {}
                    ocr_json["Pic_name"] = time_stamp_name+'.png'
                    ocr_json["text"] = iter["content"]
                    ocr_json["text_len"] = len(iter["content"])
                    json_path = '%s/%s@%sx%sx%sx%s.json' % (
                        text_image_path, time_stamp_name, int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    with open(json_path, 'w', encoding='utf-8') as file:
                        file.write(json.dumps(
                            ocr_json, indent=2, ensure_ascii=False))

                else:
                    # hash_key, text_str = ocr_tool.ocr_result_hash(roi_img)
                    # # print(key)
                    # # global key_value_dict
                    # if not hash_key in key_value_dict.keys():
                    #     key_value_dict[hash_key] = text_str
                    # folder_path =unrec_text_path + '%s/'%(hash_key)
                    # if not os.path.exists(folder_path):
                    #     os.makedirs(folder_path)

                    save_path = '%s/%s@%sx%sx%sx%s.png' % (
                        unrec_text_path, time_stamp_name, int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    
                    
                    # print(save_path)
                cv2.imwrite(save_path, roi_img)
        if(len(imgs) > 0):
            imgs_image_path = os.path.join(subimg_dir,"imgs") #'%s/imgs/' % (subimg_dir)
            if not os.path.exists(imgs_image_path):
                os.makedirs(imgs_image_path)
            for iter in imgs:
                b = [int(iter["location"][0]), int(iter["location"][1]), int(iter["location"][2]),
                     int(iter["location"][3])]
                padding_b = [max(b[0] - padding_size, 0), max(b[1] - padding_size, 0),
                             min(b[2] + padding_size, image.shape[1] - 1), min(b[3] + padding_size, image.shape[0] - 1)]
                roi_img = image[padding_b[1]: padding_b[3],
                                padding_b[0]:padding_b[2]]
                image_path = '%s/%s@%sx%sx%sx%s.png' % (
                    imgs_image_path, time_stamp_name, int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                cv2.imwrite(image_path, roi_img)


class data_tranformer:
    # def __init__(self, set_root, set_path, sub_set, auto_ocr_rec = 1, resnet_model_path = "./resnet_ocr/ResNet_testModel.pt", resnet_key_path = "./resnet_ocr/alphabet.txt"):
    def __init__(self, set_root, set_path, sub_set, auto_ocr_rec=1, std_split=False, **kwargs):
        self.dataset_root = set_root
        self.dataset_path = set_path
        self.sub_set = sub_set
        self.auto_orc_rec = auto_ocr_rec
        self.std_split = std_split
        if self.auto_orc_rec:
            #self.ocr_rec_resnet = ocr_model_pred.resnet_ocr(resnet_model_path, resnet_key_path)
            self.ocr_tool = data_ocr()
            
        self.illegal_num = 0
        self.ioverlap_num= 0
        self.valid_num = 0

    def print_stat(self):
        print(f"illegal_num: {self.illegal_num}")
        print(f"ioverlap_num: {self.ioverlap_num}")
        print(f"valid_num: {self.valid_num}")

    def reset_stat(self):
        self.illegal_num = 0
        self.ioverlap_num= 0
        self.valid_num = 0


    def parse_label_json(self, file_path, image_path=""):
        annos = None
        try:
            self.labelimg2labelme(file_path)
            print(file_path," transformed to labelme json")
        except:
            print("No Json Transformation")
        if annos is None:
            # try:
            annos = self.parse_labelme_json(file_path, image_path)
            # except:
                # print("Not labelme json")
        return annos
    
    def annos_remove_overlap(self, selected_box, annos, thresh=0.75):
        for idx, anno in enumerate(annos):
            iou_value = get_iou(selected_box, anno)
            if iou_value> thresh:
                if cal_area(selected_box) > cal_area(anno):
                    print("remove: ", anno)
                    return idx
                else:
                    print("remove: ", selected_box)
                    return -1
        return None

    def parse_labelme_json(self, file_path, image_path=""):
        annos = {}
        boxes_text = []
        boxes_icon = []
        boxes_image = []
        boxes_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
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


        for iter in ret_dic["shapes"]:
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
                self.illegal_num += 1
                continue

            if len(boxes_list)>0:
                remove_idx = self.annos_remove_overlap(b, boxes_list)
            if remove_idx is None:
                boxes_list.append(b)
            elif remove_idx == -1:
                self.ioverlap_num += 1
                continue
            else:
                remove_element = boxes_list[remove_idx]
                boxes_list.append(b)
                boxes_list.remove(remove_element)
                self.ioverlap_num += 1
            self.valid_num += 1

            label_info["location"] = b


            if cls == "text":
                if remove_idx is not None:
                    for ibboxes in boxes_text:
                        if remove_element == ibboxes["location"]:
                            boxes_text.remove(ibboxes)
                            break
                # 如果自动进行ocr识别，调用OCR识别来识别该字符串
                if self.auto_orc_rec:
                    ocr_image = image[b[1]:b[3], b[0]:b[2]]
                    text_str = self.ocr_tool.valid_ocr(ocr_image)
                    label_info["content"] = text_str
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
        annos["image"] = boxes_image
        
        return annos


    def labelimg2labelme(self, file_path):
        labelme_annos = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
        h = float(ret_dic["imgHeight"])
        w = float(ret_dic["imgWidth"])
        labelme_annos['imageHeight'] = int(h)
        labelme_annos['imageWidth'] = int(w)
        labelme_annos['imagePath'] = ret_dic["image_name"]
        shapes = []
        for iter in ret_dic["attrbutes"]:
            label = iter["label"]
            b = [int(float(iter["data"][0]) * w), int(float(iter["data"][1] * h)), int(float(iter["data"][2]) * w),
                     int(float(iter["data"][3]) * h)]
            shape = [label, b]
            shapes.append(shape)
        labelme_annos['shapes'] = shapes
        print(labelme_annos)
        save_path, json_file = os.path.split(file_path)
        write_anno(save_path, labelme_annos)
        

    # def parse_labelimg_json(self, file_path, image_path=""):

    #     annos = {}
    #     boxes_text = []
    #     boxes_icon = []
    #     boxes_image = []
    #     boxes_list = []
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         ret_dic = json.load(f)
    #         h = float(ret_dic["imgHeight"])
    #         w = float(ret_dic["imgWidth"])
    #         annos["imgHeight"] = int(h)
    #         annos["imgWidth"] = int(w)
    #         annos["imageName"] = ret_dic["image_name"]
    #         if self.auto_orc_rec:
    #             image = cv2.imread(image_path)
    #         for iter in ret_dic["attrbutes"]:
    #             cls = iter["label"]
    #             remove_idx = None
    #             label_info = {}
    #             b = [int(float(iter["data"][0]) * w), int(float(iter["data"][1] * h)), int(float(iter["data"][2]) * w),
    #                  int(float(iter["data"][3]) * h)]
    #             if (b[3] - b[1] <= 0) or (b[2] - b[0] <= 0):
    #                 print("error rect:", b)
    #                 continue
    #             if len(boxes_list)>0:
    #                 remove_idx = self.annos_remove_overlap(b, boxes_list)
    #             if remove_idx is None:
    #                 boxes_list.append(b)
    #             elif remove_idx == -1:
    #                 continue
    #             else:
    #                 boxes_list.remove(boxes_list[remove_idx])
    #             label_info["location"] = b
    #             if cls == "text":
    #                 if remove_idx is not None:
    #                     boxes_text.remove(boxes_text[remove_idx])
    #                 # 如果自动进行ocr识别，调用OCR识别来识别该字符串
    #                 if self.auto_orc_rec:
    #                     ocr_image = image[int(b[1]):int(
    #                         b[3]), int(b[0]):int(b[2])]
    #                     # print(ocr_image.shape)
    #                     text_str = self.ocr_tool.valid_ocr(ocr_image)
    #                     label_info["content"] = text_str
    #                 boxes_text.append(label_info)
    #             elif cls == "icon":
    #                 if remove_idx is not None:
    #                     boxes_icon.remove(boxes_icon[remove_idx])
    #                 boxes_icon.append(label_info)
    #             elif cls == "image":
    #                 if remove_idx is not None:
    #                     boxes_image.remove(boxes_image[remove_idx])
    #                 boxes_image.append(label_info)
    #             else:
    #                 print("bad label:", file_path)
    #     annos["icon"] = boxes_icon
    #     annos["text"] = boxes_text
    #     annos["image"] = boxes_image
    #     return annos

    def labelme_json2sa_json(self, src_path, dst_path, ocr_image=""): 
        annos = self.parse_label_json(src_path, ocr_image)
        if annos is None:
            return    
        with open(dst_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(annos, indent=2, ensure_ascii=False))
        # json.dump(annos, open(dst_path, 'w', encoding='utf-8'))

    def tranform_format(self):
        for image_set in self.sub_set:

            prefix_relative_path = self.dataset_path+'/' +image_set
            
            image_path = os.path.join(self.dataset_root, self.dataset_path, image_set , "imgs")
            
            labeljson_path = os.path.join(self.dataset_root, self.dataset_path, image_set, 'labelme_json')
            
            sajson_path = os.path.join(self.dataset_root, self.dataset_path, image_set, 'annotations')

            ensure_dir(sajson_path)
            std_path = os.path.join(self.dataset_root, self.dataset_path, image_set, 'std')
            ensure_dir(std_path)
            image_ids = os.listdir(image_path)
            for image_id in image_ids:
                # if image_id != '1607416505_img.png':
                #     continue
                name = image_id.split(".")[0]
                print("********************transform: ",
                      name, "**************************")
                # convert labelme json to sa json
                self.labelme_json2sa_json(os.path.join(labeljson_path, name + ".json"),
                                          os.path.join(
                                              sajson_path, name + ".json"),
                                          os.path.join(image_path, name + ".png"))  # win ocr
                # conver sa json to std(for task train)
                if self.std_split:
                    dest = os.path.join(std_path, name,'std.label')
                else:
                    dest = os.path.join(std_path, name+'.label')
                ensure_parent(dest)
                sa_json2std(os.path.join(sajson_path, name + ".json"),
                            dest, prefix_relative_path, self.std_split)
            
            self.print_stat()
            self.reset_stat()

        print("********************transform finished!**************************\n")

    def crop_image(self, padding_size=3):
        for image_set in self.sub_set:
            image_path = os.path.join(self.dataset_root, self.dataset_path, image_set , "imgs")
            if not os.path.exists(image_path):
                print("error error error: no images")
                return -1
            sajson_path = os.path.join(self.dataset_root, self.dataset_path, image_set, 'annotations')
            if not os.path.exists(sajson_path):
                print("error error error: no sa_json")
                return -1

            ocr_dir = os.path.join(self.dataset_root,self.dataset_path, image_set, 'crop_img')
            # build ocr image and ans path
            if not os.path.exists(ocr_dir):
                os.makedirs(ocr_dir)

            image_ids = os.listdir(image_path)
            for image_id in image_ids:
                name = image_id.split(".")[0]
                print("********************crop: ",
                      name, "**************************")
                crop_icon_text(os.path.join(sajson_path, name + ".json"),
                               os.path.join(image_path, name + ".png"), ocr_dir, self.ocr_tool, padding_size)

        print("********************crop finished!**************************\n")

    def make_classification_label_list(self, prefix_path):
        
        for image_set in self.sub_set:
            file_write = open(f"button_classification_{image_set}.list","w",encoding="utf-8")
            sub_image_root = os.path.join(self.dataset_root, self.dataset_path, image_set, 'crop_img')
            image_path_list = walk_folder_search_file(sub_image_root,".png")
            for image_path in tqdm.tqdm(image_path_list, desc="classification label making"):
                image_name = os.path.split(image_path)
                processed_image_path = image_path.replace(sub_image_root,"").replace('\\',"/")
                if "icon" in processed_image_path:
                    write_image_path = prefix_path + image_set + "/crop_img"+ processed_image_path + "\t" + "icon"
                elif "text" in processed_image_path:
                    write_image_path = prefix_path + image_set + "/crop_img"+ processed_image_path + "\t" + "text"
                else:
                    write_image_path = prefix_path + image_set + "/crop_img"+ processed_image_path + "\t" + "background"
                file_write.write(write_image_path+"\n")
        
            file_write.close()

    def move_ocr_file(self, save_folder_path):
        for image_set in self.sub_set:
            folder_path = os.path.join(self.dataset_root, self.dataset_path, image_set ,"crop_img/text/unrec")
            cluster_label_list = os.listdir(folder_path)
            for label in tqdm.tqdm(cluster_label_list):
                if len(label)<4:
                    continue
                save_folder = os.path.join(save_folder_path, image_set, label)
                ensure_dir(save_folder)
                source_folder = os.path.join(folder_path, label)
                for item_name in os.listdir(source_folder):
                    source_item = os.path.join(source_folder, item_name)
                    shutil.copy2(source_item, save_folder)
                    break

    def make_ocr_hash_map(self, root_path):
        hash_map = dict()
        json_file_list = walk_folder_search_file(root_path, ".json")
        for json_file in json_file_list:
            sys_symbol = os.sep.split()[0]
            path_name, file_name = os.path.split(json_file)
            hash_key = path_name.split(sys_symbol)[-1]
            with open(json_file, "r", encoding="utf-8")as f:
                json_dict = json.load(f)
            text_str = json_dict["text"]
            if hash_key in hash_map.keys():
                print(text_str)
                continue
            hash_map[hash_key] = text_str
        return hash_map
    
    def make_ocr_hash_map_filename(self, root_path):
        filename_map = dict()
        json_file_list = walk_folder_search_file(root_path, ".json")
        count = 0
        for json_file in tqdm.tqdm(json_file_list, desc="making filename_map"):
            path_name, file_name = os.path.split(json_file)
            hash_key = file_name.split(".")[0]
            with open(json_file, "r", encoding="utf-8")as f:
                json_dict = json.load(f)
            text_str = json_dict["text"]
            text_str = text_str.replace(" ","").replace("\"","")
            if hash_key in filename_map.keys() or text_str=="":
                count+=1
                continue
            filename_map[hash_key] = text_str
        print("filename_map Done")
        print("empty file: ", count)
        return filename_map


    def filename_map_covert(self, filename_map):
        hash_map = dict()
        for image_set in self.sub_set:
            folder_path = os.path.join(self.dataset_root, self.dataset_path, image_set, "crop_img/text/unrec")
            image_file_list = walk_folder_search_file(folder_path, ".png")
            for image_path in tqdm.tqdm(image_file_list,desc="map covert"):
                path_name, file_name = os.path.split(image_path)
                file_name_key = file_name.split(".")[0]
                if file_name_key in filename_map.keys():
                    value = filename_map[file_name_key]
                    sys_symbol = os.sep.split()[0]
                    hash_key = path_name.split(sys_symbol)[-1]
                    hash_map[hash_key] = value
        return hash_map


    def make_paddle_ocr_list_auto_labeled(self):
        write_file = open("rec_ocr_list_auto.txt","w", encoding="utf-8")
        for image_set in self.sub_set:
            folder_path = os.path.join(self.dataset_root, self.dataset_path, image_set ,"crop_img/text")
            json_label_list = walk_folder_search_file(folder_path, ".json")
            for json_label_path in tqdm.tqdm(json_label_list):
                file_name = os.path.split(json_label_path)[-1].split(".")[0]
                with open(json_label_path, "r", encoding="utf-8")as file:
                    json_dict = json.load(file)
                text_str = json_dict["text"]
                text_str = text_str.replace(" ","").replace("\"","")
                msg = "/workspace/dataset/screen_analysis/" + self.dataset_path + image_set+"/crop_img/text/"+file_name+".png\t"+text_str
                write_file.write(msg+"\n")
        write_file.close()


    def write_ocr_to_sa(self, ocr_dict):
        for image_set in self.sub_set:
            sa_json_folder_path = os.path.join(self.dataset_root, self.dataset_path, image_set,"annotations")
            sa_file_list = walk_folder_search_file(sa_json_folder_path,".json")
            for sa_json_path in tqdm.tqdm(sa_file_list,desc="write ocr to sa"):
                path_name, file_name = os.path.split(sa_json_path)
                pre_name = file_name.split(".")[0]
                with open(sa_json_path, "r", encoding="utf-8") as f:
                    sa_dict = json.load(f)
                text_list = sa_dict["text"]
                for idx, text_item in enumerate(text_list):
                    if text_item["content"] == "":
                        xmin,ymin,xmax,ymax = text_item["location"]
                        ocr_key = pre_name + "@"+str(xmin)+"x"+str(ymin)+"x"+str(xmax)+"x"+str(ymax)+".png"
                        try:
                            ocr_content = ocr_dict[ocr_key]
                            sa_dict["text"][idx]["content"]=ocr_content
                        except:
                            print(ocr_key," is not be labeled")

                with open(sa_json_path,"w", encoding="utf-8")as wf:
                    json.dump(sa_dict, wf, indent=2, ensure_ascii=False)


    def make_paddle_ocr_list_labeled(self, ocr_label_path, use_filename_map = False, write_text = True, write_sa=True, combined_list = False, **kwargs):
        if use_filename_map:
            filename_map = self.make_ocr_hash_map_filename(ocr_label_path)
            hash_map = self.filename_map_covert(filename_map)
        else:
            hash_map = self.make_ocr_hash_map(ocr_label_path)
        if write_text and combined_list:
            write_file = open("rec_ocr_list.txt","w", encoding="utf-8")
        else:
            write_file = None
        for image_set in self.sub_set:
            if write_text and write_file is None:
                write_file = open(f"rec_{image_set}_list.txt","w", encoding="utf-8")
            ocr_dict = dict()
            folder_path = os.path.join(self.dataset_root, self.dataset_path, image_set ,"crop_img/text/unrec")
            image_file_list = walk_folder_search_file(folder_path, ".png")
            for image_path in tqdm.tqdm(image_file_list):
                sys_symbol = os.sep.split()[0]
                path_name, file_name = os.path.split(image_path)
                hash_key = path_name.split(sys_symbol)[-1]
                if not hash_key in hash_map.keys():
                    continue
                text_str = hash_map[hash_key]
                ocr_dict[file_name] = text_str
                if write_file:
                    msg = "/workspace/dataset/screen_analysis/"+self.dataset_path + image_set + "/crop_img/text/unrec/"+ hash_key+"/" + file_name + "\t" + text_str
                    write_file.write(msg+"\n")
            if write_text and not combined_list:
                write_file.close()
        if write_text and combined_list:
            write_file.close()

        if write_sa:
            self.write_ocr_to_sa(ocr_dict)



    # def tranform_format_ori(self):
    #     for image_set in self.sub_set:
    #         if not os.path.exists('%s/%s/%s/' % (self.dataset_root, self.dataset_path, image_set)):
    #             os.makedirs('%s/%s/%s/' %
    #                         (self.dataset_root, self.dataset_path, image_set))

    #         image_path = '%s/%s/%s/imgs/' % (self.dataset_root,
    #                                          self.dataset_path, image_set)
    #         if not os.path.exists(image_path):
    #             os.makedirs(image_path)

    #         labeljson_path = '%s/%s/%s/labelme_json/' % (
    #             self.dataset_root, self.dataset_path, image_set)
    #         if not os.path.exists(labeljson_path):
    #             os.makedirs(labeljson_path)

    #         sajson_path = '%s/%s/%s/annotations/' % (
    #             self.dataset_root, self.dataset_path, image_set)
    #         if not os.path.exists(sajson_path):
    #             os.makedirs(sajson_path)

    #         std_path = '%s/%s/%s/std/' % (self.dataset_root,
    #                                       self.dataset_path, image_set)
    #         if not os.path.exists(std_path):
    #             os.makedirs(std_path)

    #         image_ids = os.listdir(image_path)
    #         for image_id in image_ids:
    #             name = image_id.split(".")[0]
    #             print("********************transform: ",
    #                   name, "**************************")
    #             # convert labelme json to sa json
    #             self.labelme_json2sa_json(os.path.join(labeljson_path, name + ".json"),
    #                                       os.path.join(
    #                                           sajson_path, name + ".json"),
    #                                       os.path.join(image_path, name + ".png"))  # win ocr
    #             # conver sa json to std(for task train)
    #             sa_json2std(os.path.join(sajson_path, name + ".json"),
    #                         os.path.join(std_path, name))
    #     print("********************transform finished!**************************\n")

    # def tranform_format_postprocess(self):
    #     # pass_file = True
    #     for image_set in self.sub_set:
    #         # if image_set == "1612679401_img":
    #         #     pass_file = False
    #         # if pass_file:
    #         #     continue
    #         if not os.path.exists('%s/%s/%s/' % (self.dataset_root, self.dataset_path, image_set)):
    #             os.makedirs('%s/%s/%s/' %
    #                         (self.dataset_root, self.dataset_path, image_set))

    #         image_path = '%s/%s/%s/imgs/' % (self.dataset_root,
    #                                          self.dataset_path, image_set)
    #         if not os.path.exists(image_path):
    #             os.makedirs(image_path)
    #         # origin_image_path = '%s/%s/%s/%s.png' % (self.dataset_root,
    #         #                                  self.dataset_path, image_set, image_set)
    #         origin_image_path = '%s/%s/%s/img.png' % (self.dataset_root,
    #                                          self.dataset_path, image_set)
            

    #         labeljson_path = '%s/%s/%s/labelme_json/' % (self.dataset_root, self.dataset_path, image_set)
    #         if not os.path.exists(labeljson_path):
    #             os.makedirs(labeljson_path)

    #         origin_labeljson_path =  '%s/%s/%s/img.json' % (self.dataset_root, self.dataset_path, image_set)
    #         renamed_labeljson_path = '%s/%s/%s/%s.json' % (self.dataset_root, self.dataset_path, image_set, image_set)
    #         try:
    #             os.rename(origin_labeljson_path, renamed_labeljson_path)
    #         except:
    #             pass
    #         origin_labeljson_path =  '%s/%s/%s/label.json' % (self.dataset_root, self.dataset_path, image_set)
    #         try:
    #             os.rename(origin_labeljson_path, renamed_labeljson_path)
    #         except:
    #             pass

    #         sajson_path = '%s/%s/%s/annotations/' % (
    #             self.dataset_root, self.dataset_path, image_set)
    #         if not os.path.exists(sajson_path):
    #             os.makedirs(sajson_path)

    #         std_path = '%s/%s/%s/std/' % (self.dataset_root,
    #                                       self.dataset_path, image_set)
    #         if not os.path.exists(std_path):
    #             os.makedirs(std_path)
            
    #         origin_stdlabel_path =  '%s/%s/%s/std.label' % (self.dataset_root, self.dataset_path, image_set)
            
    #         try:
    #             shutil.move(origin_image_path, image_path)
    #         except:
    #             print("image moved")
    #         # try:
    #         #     shutil.move(balanced_image_path, image_path)
    #         # except:
    #         #     pass 
    #         try:
    #             shutil.move(renamed_labeljson_path, labeljson_path)
    #         except:
    #             pass
    #         try:
    #             shutil.move(origin_stdlabel_path, std_path)
    #         except:
    #             print("Move Done")

          
    #         print("********************transform: ", image_set, "**************************")
    #             # convert labelme json to sa json
            
    #         self.labelme_json2sa_json(os.path.join(labeljson_path, image_set + ".json"),
    #                                     os.path.join(
    #                                         sajson_path, image_set + ".json"),
    #                                     os.path.join(image_path, "img.png"))  # win ocr
    #         # conver sa json to std(for task train)
    #         sa_json2std(os.path.join(sajson_path, image_set + ".json"),
    #                     os.path.join(std_path, image_set))
    #     print("********************transform finished!**************************\n")

    # def crop_image_postprocess(self, padding_size=3):
    #     for image_set in self.sub_set:
        
    #         image_path = '%s/%s/%s/imgs/' % (self.dataset_root,
    #                                          self.dataset_path, image_set)
    #         if not os.path.exists(image_path):
    #             print("error error error: no images")
    #             return -1
    #         sajson_path = '%s/%s/%s/annotations/' % (
    #             self.dataset_root, self.dataset_path, image_set)
    #         if not os.path.exists(sajson_path):
    #             print("error error error: no sa_json")
    #             return -1

    #         ocr_dir = '%s/%s/%s/crop_img/' % (self.dataset_root,
    #                                           self.dataset_path, image_set)
    #         # build ocr image and ans path
    #         if os.path.exists(ocr_dir):
    #             print("del", ocr_dir)
    #             shutil.rmtree(ocr_dir)
    #         # if not os.path.exists(ocr_dir):
    #         os.makedirs(ocr_dir)
    #         # else:
    #         print("********************crop: ",
    #                 image_set, "**************************")
    #         # print(os.path.join(sajson_path, image_set + ".json"))
    #         crop_icon_text(os.path.join(sajson_path, image_set + ".json"),
    #                         os.path.join(image_path, "img.png"), ocr_dir, self.ocr_tool, padding_size)

    #     print("********************crop finished!**************************\n")


class data_detect_transform:

    def __init__(self, debug_level=1, front_image_path = "logo.png"):
        self.debug_level = debug_level
        self.front_image_path = front_image_path
        self.front_image = cv2.imread(front_image_path)

    def preprocess(self, input_folder, output_folder="DEFAULT_MAGIC_STR"):
        input_folders = [os.path.join(input_folder, t)
                         for t in os.listdir(input_folder) if t.count("_") < 3]
        input_folders = [input_folder]
        task_params = []
        for input_folder in input_folders:
            if output_folder == "DEFAULT_MAGIC_STR":
                output_folder = input_folder + "_template_match2"
            if os.path.exists(output_folder):
                rmtree(output_folder)
            ensure_dir(output_folder)
            task_params.append(
                [input_folder, output_folder, "uia", self.debug_level])
        r = list(tqdm.tqdm(starmap(template_match_pre, task_params)))
        # with Pool(processes=cpu_count()) as pool:
            
        #     pool.close()
        #     pool.join()


    def add_weight(self, bottom_image):
        h, w = bottom_image.shape[0], bottom_image.shape[1]
        background = np.zeros(bottom_image.shape).astype(np.uint8)+255
        for i in range(0, h-self.front_image.shape[0]-100, self.front_image.shape[0]+100):
            for j in range(0, w - self.front_image.shape[1]-100, self.front_image.shape[1]+100):
                background[i:i+self.front_image.shape[0], j:j+self.front_image.shape[1]] = self.front_image
        overlapping = cv2.addWeighted(bottom_image, 0.9, background, 0.1, 0)
        return overlapping


    def add_weight_to_img(self, source_folder, output_folder="DEFAULT_MAGIC_STR", image_type = "png"):
        if output_folder == "DEFAULT_MAGIC_STR":
            output_folder = source_folder + "_add_weight"
        print(output_folder)
        folder_list = os.listdir(source_folder)
        for image_name in tqdm.tqdm(folder_list):
            if image_name.split('.')[-1]==image_type:
                image_path = os.path.join(source_folder, image_name)
                # print(image_path)
                # image_path = os.path.join(image_folder_name,image_name)
                bottom_image = cv2.imread(image_path)
                overlapping = self.add_weight(bottom_image)
                ensure_dir(output_folder)
                image_save_path = os.path.join(output_folder,image_name)
                cv2.imwrite(image_save_path, overlapping)


    def postprocess(self, source_folder, anno_file, std_dataset_folder, output_folder="DEFAULT_MAGIC_STR", label_type="detect", output_type="std", sa_api = True):
        match_folder = anno_file
        source_folders = [pathj(source_folder, t) for t in os.listdir(
            source_folder) if "_" not in t]
        source_folders = [source_folder]
        task_params = []
        for source_folder in source_folders:
            software_name = os.path.split(source_folder)[-1]
            if output_folder == "DEFAULT_MAGIC_STR":
                output_folder = source_folder + "_template_match_post_test"
            if os.path.exists(output_folder):
                rmtree(output_folder)
            ensure_dir(output_folder)
            task_params.append([source_folder, match_folder, anno_file,
                                output_folder, output_type, std_dataset_folder, label_type, self.debug_level, sa_api])
        r = list(tqdm.tqdm(starmap(template_match_post, task_params)))
        # with Pool(processes=cpu_count()) as pool:
            
        #     pool.close()
        #     pool.join()


class data_ocr:

    def __init__(self,resnet_model_path="./ocr/OCR/model/CRNN/recognition.onnx", resnet_key_path="./ocr/OCR/model/CRNN/mmocr_charset"):
        
        def get_keyword_str(file):
            with open(file, "r") as charFile:
                keyForONNX = charFile.readlines()
            charFile.close()
            keyForONNX = [i[:-1] for i in keyForONNX]
            return keyForONNX 
        self.ocr_onnx = onnxruntime.InferenceSession(resnet_model_path)
        self.keyForONNX = get_keyword_str(resnet_key_path)
        self.paddleOcrModel = PaddleOCR(rec_model_dir='./ocr/OCR/model/PadleOCR/ch_ppocr_server_v2.0_rec_infer', use_angle_cls=True, use_gpu=False)

    def valid_ocr(self, ocr_image):
        #crnnnet_res = self.ocr_rec_resnet.ocr_result_image(ocr_image).strip()
        crnnnet_res, crnnnet_res_score = PaddleOCR_CRNNOCR.CRNN_Predict(ocr_image, self.ocr_onnx, self.keyForONNX)
        crnnnet_res = crnnnet_res.replace("（", "(")
        crnnnet_res = crnnnet_res.replace("）", ")")
        # paddle paddle  result
        paddle_res, paddle_res_score = PaddleOCR_CRNNOCR.PaddleOCR_Predict(ocr_image,self.paddleOcrModel)
        paddle_res = paddle_res.replace("（", "(")
        paddle_res = paddle_res.replace("）", ")")
        print(" ocr content: ", len(crnnnet_res), " ", crnnnet_res,
              " vs ", len(paddle_res), " ", paddle_res)
        if crnnnet_res == paddle_res:
            return paddle_res
        else:
            if paddle_res_score > 0.9:
                return paddle_res
            return ""

    def ocr_result_hash(self, img, cluster_model='Paddle'):

        if cluster_model == "Paddle":
            paddle_res = paddle_ocr_rec_eval.text_sys.process([img])
            text_str = paddle_res[0][0].strip()
            hash_key = hash(text_str)

        elif cluster_model == "Densectc":
            text_str, score = self.ocr_onnx.predict(img)
            hash_key = hash(resnet_res)

        else:
            return ""
        
        return hash_key, text_str


def get_third_party_label_img(root_path, sub_set, save_path):
    
    for src_name in sub_set:
        src_path = os.path.join(root_path, src_name,"crop_img","text","unrec")
        dst_path = os.path.join(save_path, src_name)
        ensure_dir(save_path)
        remove_dir(dst_path)
        shutil.copytree(src_path, dst_path)

class update_dataset:
    def __init__(self,label_path, sub_set, sa_update, sa_root_path, labelme_update, labelme_root_path):
        self.label_path = label_path
        self.sub_set = sub_set
        self.sa_update = sa_update
        self.sa_root_path = sa_root_path
        self.labelme_update = labelme_update
        self.labelme_root_path = labelme_root_path
    
    def update(self):
        if self.sa_update:
            self.update_dataset_sa()
        if self.labelme_update:
            self.update_dataset_labelme()

    def update_dataset_sa(self):
        t =1
    
    def update_dataset_labelme(self):
        for set_name in self.sub_set:
            set_path = os.path.join(self.label_path, set_name, )
            json_list = os.listdir(set_path)

            for json_name in json_list:
                json_path = os.path.join(set_path, json_name)
                with open(json_path,'r') as json_file:
                    json_content = json_file.readlines()
                    content = json.loads(json_content[0])
                    #判别元素是否为text文本，如果是icon则修改对应labelme中的标签
                    if(content["type"] == "icon"):
                        change_path = os.path.join(self.labelme_root_path, set_name, json_name.split("@")[0])
                        
                        #修改std.label文件
                        std_change_path = os.path.join(change_path, "std.label")
                        std_file = open(std_change_path, 'r')
                        std_content = std_file.readlines()
                        search_content = "text\t"+json_path.split("@")[1].replace("x","\t").replace(".json","\n")
                        replace_content = "icon\t"+json_path.split("@")[1].replace("x","\t").replace(".json","\n")
                        save_content=[replace_content if i == search_content else i for i in std_content]
                        save_file = open(std_change_path, 'w')
                        save_file.writelines(save_content)
                        save_file.close()

                        #修改label.json
                        labelme_change_path = os.path.join(change_path, "label.json")
                        labelme_file = open(labelme_change_path, 'r')
                        labelme_content = json.load(labelme_file)
                        element_location = [int(i) for i in json_path.split("@")[1].split(".")[0].split("x")]
                        for i in range(len(labelme_content["shapes"])):
                            if labelme_content["shapes"][i]["label"] == "text" and int(labelme_content["shapes"][i]["points"][0][0]) == element_location[0] and int(labelme_content["shapes"][i]["points"][0][1]) == element_location[1] and int(labelme_content["shapes"][i]["points"][1][0]) == element_location[2] and int(labelme_content["shapes"][i]["points"][1][1]) == element_location[3]:
                                labelme_content["shapes"][i]["label"] = "icon"
                                print(labelme_content["shapes"][i])
                                break
                            
                        save_labelme_file = open(labelme_change_path, 'w')
                        save_content = json.dumps(labelme_content,indent=2)
                        save_labelme_file.write(save_content)
                        save_labelme_file.close()

                    json_file.close()
                    # 'C:\\Users\\lei.gao\\Desktop\\ocr\\result_55907\\0003_excel_07pro\\1607498344_img@2523x1404x2530x1414.json'