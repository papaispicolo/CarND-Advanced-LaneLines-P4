#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=W0312,

"""
Util methods for image data browsing, etc.
"""
import sys
import cv2
import base64
import pickle
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import fnmatch
from base64 import b64encode, b64decode, decodestring
from subprocess import Popen, PIPE
from datetime import datetime
from io import open, StringIO, BytesIO
from sys import stdin, stdout, stderr
from itertools import groupby
from operator import itemgetter
from shlex import split
from os import devnull, errno
from json import loads,dumps


def count_dataset(path_datalist,label="*") :
    num_lines = sum(1 if line.strip().split(" ")[1] == label else \
                      (1 if label == "*" else 0) for line in open(path_datalist))
    return num_lines


def count_dataset_of_labels(path_datalist,labels=[]) :
    """
    labels to count
    """
    cnt_label = [ count_dataset(path_datalist,str(i)) for i in  labels ]
    total = sum(i for i in cnt_label)
    print ("total", total)

    return total


def get_sampled_dataset(basedir,path_datalist,labels=[],\
                         sampling_rate=0.001) :
    """
    labels to count
    """
    str_labels = [str(i) for i in labels]
    data = [ line.strip().split(" ")[0] for line in open(path_datalist) \
             if line.strip().split(" ")[1] in str_labels ]
    num_elem = len(data)
    ds_size= int(num_elem * sampling_rate)
    print ("# sample : {}  sampling_rate : {}  # of data : {}".format(ds_size,sampling_rate,num_elem))
    sampled_data = random.sample(data, ds_size)

    return sampled_data

def show_sampled_dataset(basedir,path_datalist,labels=[],\
                         sampling_rate=0.001,thumb_w=64,thumb_h=128,\
                         title="") :
    """
    labels to count
    """
    str_labels = [str(i) for i in labels]
    data = [ line.strip().split(" ")[0] for line in open(path_datalist) \
             if line.strip().split(" ")[1] in str_labels ]
    num_elem = len(data)
    ds_size= int(num_elem * sampling_rate)
    print ("# sample : {}  sampling_rate : {}  # of data : {}".format(ds_size,sampling_rate,num_elem))
    sampled_data = random.sample(data, ds_size)

    n = math.sqrt(ds_size)
    c = int(n)  + 2 # columns and rows of thumbnail matrix
    r = int(ds_size / c )
    r = r + 1 if ds_size % c > 0 else r

    w = int(c * thumb_w)
    h = int(r * thumb_h)

    # create an empty image
    size = h, w, 3
    print ("size = ", size)
    vis_map = np.zeros(size, dtype=np.uint8)

    # Make w as 10  - 10 = w : v : h
    vr = 10 * h / w

    plt.figure(figsize=(10,vr))

    for i in range(0,r):
        for j in range(0,c):
            idx = i*c + j
            try :
                img = cv2.imread(basedir + sampled_data[idx])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img_thumb = cv2.resize(img,(thumb_w, thumb_h), interpolation = cv2.INTER_CUBIC)
                thumb_y = i * thumb_h
                thumb_x = j * thumb_w
                vis_map[thumb_y:thumb_y+thumb_h,thumb_x:thumb_x+thumb_w] = img_thumb
            except :
                pass

    plt.title(title)
    plt.imshow(vis_map,interpolation='nearest', aspect='auto'),plt.show()

    return sampled_data


def gen_find(filepat,top,parentaslabel=False,labels=[]):
    """k
    Usage: pyfiles = gen_find("*.py","/")
    logs = gen_find("access-log*","/usr/www/")
    """
    topdir = top if top.endswith("/") else top+"/"

    for parentdir, dirlist, filelist in os.walk(topdir) :
        for fname in fnmatch.filter(filelist,filepat):
            t = os.path.join(parentdir,fname)
            label = "-"
            if parentaslabel :
                try :
                    label = parentdir[len(topdir):].split("/")[0]
                except :
                    pass

            if not os.path.islink(t) :
                yield label,t


def count_dataset_from_dir(path_data_dir,filepat="*", \
                           parentaslabel=False,\
                           labels=[]) :
    """
    count data
    """
    data = [ f for lbl, f in gen_find(filepat,path_data_dir,parentaslabel,labels) \
             if not parentaslabel or ( parentaslabel and lbl in labels) ]

    return len(data)


def get_sampled_dataset_from_dir(path_data_dir,filepat="*", \
                                  parentaslabel=False,\
                                  labels=[],\
                                  sampling_rate=0.001) :
    """
    labels to count
    """
    data = [ f for lbl, f in gen_find(filepat,path_data_dir,parentaslabel,labels) \
             if not parentaslabel or ( parentaslabel and lbl in labels) ]
    num_elem = len(data)
    ds_size= int(num_elem * sampling_rate)
    print ("# sample : {}  sampling_rate : {}  # of data : {}".format(ds_size,sampling_rate,num_elem))
    sampled_data = random.sample(data, ds_size)

    return sampled_data


def show_sampled_dataset_from_dir(path_data_dir,filepat="*", \
                                  parentaslabel=False,\
                                  labels=[],\
                                  sampling_rate=0.001,thumb_w=64,thumb_h=128,\
                                  title="") :
    """
    labels to count
    """
    data = [ f for lbl, f in gen_find(filepat,path_data_dir,parentaslabel,labels) \
             if not parentaslabel or ( parentaslabel and lbl in labels) ]
    num_elem = len(data)
    ds_size= int(num_elem * sampling_rate)
    print ("# sample : {}  sampling_rate : {}  # of data : {}".format(ds_size,sampling_rate,num_elem))
    sampled_data = random.sample(data, ds_size)

    n = math.sqrt(ds_size)
    c = int(n)  + 2 # columns and rows of thumbnail matrix
    r = int(ds_size / c )
    r = r + 1 if ds_size % c > 0 else r

    w = int(c * thumb_w)
    h = int(r * thumb_h)

    # create an empty image
    size = h, w, 3
    print ("size = ", size)
    vis_map = np.zeros(size, dtype=np.uint8)

    # Make w as 10  - 10 = w : v : h
    vr = 10 * h / w

    plt.figure(figsize=(10,vr))

    for i in range(0,r):
        for j in range(0,c):
            idx = i*c + j
            try :
                img = cv2.imread(sampled_data[idx])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img_thumb = cv2.resize(img,(thumb_w, thumb_h), interpolation = cv2.INTER_CUBIC)
                thumb_y = i * thumb_h
                thumb_x = j * thumb_w
                vis_map[thumb_y:thumb_y+thumb_h,thumb_x:thumb_x+thumb_w] = img_thumb
            except :
                pass

    plt.title(title)
    plt.imshow(vis_map,interpolation='nearest', aspect='auto'),plt.show()

    return sampled_data


def show_histogram_of_labels(path_datalist,labels=[],label_name=None) :
    """
    labels to count
    """
    label_count = [ count_dataset(path_datalist,str(i)) for i in labels ]

    index = labels if label_name is None else label_name

    df = pd.Series(label_count, index=index)
    df.plot.bar(title="histogram of data per labels")

    return df


def show_histogram_of_labels_from_dir(path_data_dir,filepat="*", \
                                      labels=[],\
                                      label_name=None) :
    """
    It assumes the level 1 directory from the path_data_dir to be a label
    """
    label_count = [ count_dataset_from_dir(os.path.join(path_data_dir,lbl),filepat) for lbl in labels ]

    index = labels if label_name is None else label_name

    df = pd.Series(label_count, index=index)
    df.plot.bar(title="histogram of data per labels")

    return df


def display_img_info(img) :
    """
    display image info

    img = image numpyarray
    """
    print (img.shape, img.ndim, img.dtype.name, type(img))


def show_image(basedir,path,w=None,h=None) :
    """
    Display image of given path
    resize image when w and h are set
    """
    img_path = path if basedir is None else "{}{}".format(basedir,path)
    img1 = cv2.imread(img_path)
    if w and h :
        img1 = cv2.resize(img1,(w,h), interpolation = cv2.INTER_CUBIC)

    h, w = img1.shape[:2]
    # Make w as 10
    # 10 = w : v : h
    vr = 10 * h / w

    plt.figure(figsize=(10,vr))
    plt.imshow(img1,interpolation='nearest', aspect='auto'),plt.show()
    return img1


def show_cvimage(img,title="") :
    """
    Display image of given path
    resize image when w and h are set
    """
    h, w = img.shape[:2]
    # Make w as 10
    # 10 = w : v : h
    vr = 10 * h / w

    plt.figure(figsize=(10,vr))
    plt.title(title)
    plt.imshow(img,interpolation='nearest', aspect='auto'),plt.show()
    return img



def show_patch(basedir,path,patch_x,patch_y,patch_w,patch_h) :
    """
    Display image patch of given path
    """
    img_path = path if basedir is None else "{}{}".format(basedir,path)
    img1 = cv2.imread(img_path)
    img1 = img1[patch_y:patch_y+patch_h,patch_x:patch_x+patch_w]

    h, w = img1.shape[:2]
    # Make w as 5
    # 5 = w : v : h
    vr = 5 * h / w

    plt.figure(figsize=(5,vr))
    plt.imshow(img1,interpolation='nearest', aspect='auto'),plt.show()
    return img1


def average_img(basedir,path_datalist,labels=[],\
                sampling_rate=0.001,\
                title="") :
    """
	create and visualize average image of given dataset
    """
    str_labels = [str(i) for i in labels]
    data = [ line.strip().split(" ")[0] for line in open(path_datalist) \
             if line.strip().split(" ")[1] in str_labels ]
    num_elem = len(data)
    ds_size= int(num_elem * sampling_rate)
    print ("# sample : {}  sampling_rate : {}  # of data : {}".format(ds_size,sampling_rate,num_elem))
    sampled_data = random.sample(data, ds_size)

    print (sampled_data[0])
    img_avg = cv2.imread(os.path.join(basedir,sampled_data[0]))
    h, w = img_avg.shape[:2]

    nSum = 1
    for i in sampled_data[1:] :
        imga = cv2.resize(cv2.imread(os.path.join(basedir,i)),\
                          (w, h), interpolation = cv2.INTER_CUBIC)
        weight_avg = float(nSum)/float(nSum+1)
        weight_a = float(1)/float(nSum+1)
        img_avg = cv2.addWeighted(img_avg,weight_avg,imga,weight_a,0)
        #print ("Weight_avg : {} + Weight_a : {} = Total Weight {} " \
        #       .format(weight_avg,weight_a,weight_avg+weight_a))
        nSum+=1

    # Make w as 10  - 10 = w : v : h
    vr = 10 * h / w

    plt.figure(figsize=(10,vr))
    plt.title(title)
    plt.imshow(img_avg,interpolation='nearest', aspect='auto'),plt.show()

    return img_avg


def average_img_from_dir(path_data_dir,filepat="*", \
                parentaslabel=False,\
                labels=[],\
                sampling_rate=0.001,\
                title="average image") :
    """
	create and visualize average image of given dataset
	dataset_path  = path to dataset
	sampling_rate = sampling ratio to visualize value in [0,1]
	seed = seed number to use for random value generation

	return
	average_image = average image of the given dataset
    """
    data = [ f for lbl, f in gen_find(filepat,path_data_dir,parentaslabel,labels) \
            if not parentaslabel or ( parentaslabel and lbl in labels) ]
    num_elem = len(data)
    ds_size= int(num_elem * sampling_rate)
    print ("# sample : {}  sampling_rate : {}  # of data : {}".format(ds_size,sampling_rate,num_elem))
    sampled_data = random.sample(data, ds_size)

    img_avg = cv2.imread(sampled_data[0])
    h, w = img_avg.shape[:2]

    nSum = 1
    for i in sampled_data[1:] :
        imga = cv2.resize(cv2.imread(i),(w, h), interpolation = cv2.INTER_CUBIC)
        weight_avg = float(nSum)/float(nSum+1)
        weight_a = float(1)/float(nSum+1)
        img_avg = cv2.addWeighted(img_avg,weight_avg,imga,weight_a,0)
        #print ("Weight_avg : {} + Weight_a : {} = Total Weight {} " \
        #       .format(weight_avg,weight_a,weight_avg+weight_a))
        nSum+=1

    # Make w as 10  - 10 = w : v : h
    vr = 10 * h / w

    plt.figure(figsize=(10,vr))
    plt.title(title)
    plt.imshow(img_avg,interpolation='nearest', aspect='auto'),plt.show()

    return img_avg
