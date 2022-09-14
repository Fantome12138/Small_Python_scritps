# -*- coding: utf-8 -*-

import os
from easydict import EasyDict

Cfg = EasyDict()

Cfg.names = ['server']
# 由于原对象的名字太长，绘制在图片上显得很杂乱，所以将名字简写。
Cfg.textnames = ['server']

Cfg.device = '0,1'

# manual
Cfg.origimgs_filepath = '../data_test/JPEGImages_manual'
Cfg.testimgs_filepath = '../data_test/JPEGImages_manual'
Cfg.eval_classtxt_path = '../data_test/class_txt_manual/'
Cfg.eval_Annotations_path = '../data_test/Annotations_manual'
Cfg.eval_imgs_name_txt = '../data_test/imgs_name_manual.txt'
Cfg.cachedir = '../data_test/cachedir_manual/'
Cfg.prediction_path = '../data_test/predictions_manual'

# mAP_line cachedir
Cfg.systhesis_valid_cachedir = '../data_test/cachedir_systhesis_valid/'
Cfg.manual_cachedir = '../data_test/cachedir_manual/'
