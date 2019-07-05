#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Harry
# Time: 2019/6/29


import glob
import cv2


path = ''
out = ''


temp = glob.glob(path + '/*.png')
temp.sort()

for i, pat in enumerate(temp):
    img = cv2.imread(pat)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray_3 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(out+pat[-21:], img_gray_3)
