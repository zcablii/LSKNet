import os
import cv2
from xml.dom.minidom import parse
from tqdm import tqdm
import sys
import random

def solve_xml(src, tar):
    domTree = parse(src)
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName("objects")[0].getElementsByTagName("object")
    box_list=[]
    for obj in objects:
        name=obj.getElementsByTagName("possibleresult")[0].getElementsByTagName("name")[0].childNodes[0].data
        points=obj.getElementsByTagName("points")[0].getElementsByTagName("point")
        bbox=[]
        for point in points[:4]:
            x=point.childNodes[0].data.split(",")[0]
            y=point.childNodes[0].data.split(",")[1]
            bbox.append(float(x))
            bbox.append(float(y))
        box_list.append({"name":name, "bbox":bbox})

    file=open(tar,'w')
    print("imagesource:GoogleEarth",file=file)
    print("gsd:0.0",file=file)
    for box in box_list:
        ss=""
        for f in box["bbox"]:
            ss+=str(f)+" "
        name=  box["name"]
        name = name.replace(" ", "_")
        ss+=name+" 0"
        print(ss,file=file)
    file.close()

def fair_to_dota(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "images"), exist_ok=True)

    tasks = []
    for root, dirs, files in os.walk(os.path.join(in_path, "images")):
        for f in files:
            src=os.path.join(root, f)
            tar="P"+f[:-4].zfill(4)+".png"
            tar=os.path.join(out_path,"images", tar)
            tasks.append((src, tar))
    print("processing images")
    for task in tqdm(tasks):
        file = cv2.imread(task[0], 1)
        cv2.imwrite(task[1], file)

    if (os.path.exists(os.path.join(in_path, "labelXml"))):
        os.makedirs(os.path.join(out_path, "labelTxt"), exist_ok=True)
        tasks = []
        for root, dirs, files in os.walk(os.path.join(in_path, "labelXml")):
            for f in files:
                src=os.path.join(root, f)
                tar="P"+f[:-4].zfill(4)+".txt"
                tar=os.path.join(out_path,"labelTxt", tar)
                tasks.append((src, tar))
        print("processing labels")
        for task in tqdm(tasks):
            solve_xml(task[0], task[1])


if __name__ == '__main__':
    src = sys.argv[1]
    tar = sys.argv[2]
    fair_to_dota(src, tar)
