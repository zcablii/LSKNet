import pickle
import os, sys
import cv2
from xml.dom.minidom import parse
from tqdm import tqdm
def pick_res(src_path, images_dir, keep_underline=False):
    res={}
    imgs = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if not f.endswith(".png"):
                continue
            name = f.split("__")[0]
            res[name]=[]
    for root, dirs, files in os.walk(src_path):
        for f in files:
            if f[-3:] == 'zip':
                continue
            src=os.path.join(root, f)
            if keep_underline:
                cls = f[:-4]
            else:
                cls=f[6:-4].replace("_"," ")
            with open(src, "r") as ff:
                tot_data = ff.read().split("\n")
                for data in tot_data:
                    if len(data)<5:
                        continue
                    data = data[:-1].split(" ")
                    box=[]
                    for i in range(2, len(data)):
                        box.append(float(data[i]))
                    if not data[0] in res:
                        assert(False)
                        res[data[0]]=[]
                    res[data[0]].append({"cls":cls, "p":float(data[1]), "box":box})
    return res

def dota_to_fair(src_path, tar_path, images_dir):
    """
    src_path:   dota txt format result path. e.g. ../../../worl_dirs/results/orcnn_res50
    tar_path:   the file path to save the xml result
    images_dir: the folder path of test images
    """
    data=pick_res(src_path, images_dir)
    head="""<?xml version="1.0" encoding="utf-8"?>
    <annotation>
        <source>
        <filename>placeholder_filename</filename>
        <origin>GF2/GF3</origin>
        </source>
        <research>
            <version>4.0</version>
            <provider>placeholder_affiliation</provider>
            <author>placeholder_authorname</author>
            <pluginname>placeholder_direction</pluginname>
            <pluginclass>placeholder_suject</pluginclass>
            <time>2020-07-2020-11</time>
        </research>
        <size>
            <width>placeholder_width</width>
            <height>placeholder_height</height>
            <depth>placeholder_depth</depth>
        </size>
        <objects>
    """
    obj_str="""        <object>
                <coordinate>pixel</coordinate>
                <type>rectangle</type>
                <description>None</description>
                <possibleresult>
                    <name>palceholder_cls</name>                
                    <probability>palceholder_prob</probability>
                </possibleresult>
                <points>  
                    <point>palceholder_coord0</point>
                    <point>palceholder_coord1</point>
                    <point>palceholder_coord2</point>
                    <point>palceholder_coord3</point>
                    <point>palceholder_coord0</point>
                </points>
            </object>
    """
    tail="""    </objects>
    </annotation>
    """

    os.makedirs(tar_path, exist_ok=True)
    for i in tqdm(data):
        # print(i)
        out_xml=head.replace("placeholder_filename",str(int(i[1:]))+".tif")
        out_xml=out_xml.replace("placeholder_width",str(1000))
        out_xml=out_xml.replace("placeholder_height",str(1000))
        out_xml=out_xml.replace("placeholder_depth",str(3))
        for obj in data[i]:
            obj_xml=obj_str.replace("palceholder_cls", obj["cls"])
            obj_xml=obj_xml.replace("palceholder_prob", str(obj["p"]))
            obj_xml=obj_xml.replace("palceholder_coord0", str(obj["box"][0])+", "+str(obj["box"][1]))
            obj_xml=obj_xml.replace("palceholder_coord1", str(obj["box"][2])+", "+str(obj["box"][3]))
            obj_xml=obj_xml.replace("palceholder_coord2", str(obj["box"][4])+", "+str(obj["box"][5]))
            obj_xml=obj_xml.replace("palceholder_coord3", str(obj["box"][6])+", "+str(obj["box"][7]))
            out_xml+=obj_xml
        out_xml+=tail
        with open(tar_path+"/"+str(int(i[1:]))+".xml", 'w') as f:
            f.write(out_xml)




if __name__ == '__main__':
    src = sys.argv[1]
    tar = sys.argv[2]
    images_dir = sys.argv[3]
    dota_to_fair(src, tar, images_dir)

# python tools/data/fair/dota_to_fair.py ./work_dirs/results/orcnn_lsk5b1_ema ./fair_res/test /opt/data/private/LYX/data/split_ms_dota/test/images
