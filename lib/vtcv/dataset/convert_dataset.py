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
import numpy as np
import matplotlib.pyplot as plt
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


def base64_to_image(img_s,width,height,nch):
    """
    decode image from base64 string
    """
    r = decodestring(img_s)
    i = np.frombuffer(r, dtype=np.uint8)
    img =  i.reshape(height,width,nch)
    return img

