import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import math




class SceneLoader(data.Dataset):
	def __init__(self, args, dataset, converter):
		self.args = args
		self.root = args.root
		self.dataset = dataset
		self.img_lists = []
		self.gts = []
		self.lexicon50 = []
		self.lexicon1k = []
		self.full_lexicon = []
		if self.dataset == 'ic03':
			self.root = self.root + '/ic03/' 
			for line in open(self.root + 'icdar2003_lexicon_50.txt').readlines():
				parts = line.strip().split(' ')
				self.img_lists.append(self.root + parts[0])
				self.gts.append(parts[1])
				self.lexicon50.append(parts[2:])
				self.lexicon1k.append([])
			for line in open(self.root + 'icdar2003_full_lexicon.txt').readlines():
				self.full_lexicon.append(line.strip().split())

		elif self.dataset == 'ic13':
			self.root = self.root + '/ic13/'
			for line in open(self.root + 'test_groundtruth.txt').readlines():
				parts = line.strip().split(' ')
				self.img_lists.append(self.root + parts[0])
				self.gts.append(parts[1])
				self.lexicon50.append([])
				self.lexicon1k.append([])

		elif self.dataset == 'iiit5k':
			self.root = self.root + '/iiit5k/'
			for line in open(self.root + 'iiit5k_lexicon_50.txt').readlines():
				parts = line.strip().split(' ')
				self.img_lists.append(self.root + parts[0])
				self.gts.append(parts[1])
				self.lexicon50.append(parts[2:])
			for line in open(self.root + 'iiit5k_lexicon_1k.txt').readlines():
				parts = line.strip().split(' ')
				self.lexicon1k.append(parts[2:])

		elif self.dataset == 'svt':
			self.root = self.root + '/svt/'
			for line in open(self.root + 'test_gt_lexicon.txt').readlines():
				parts = line.strip().split(' ')
				self.img_lists.append(self.root + parts[0])
				self.gts.append(parts[1])
				self.lexicon50.append(parts[2:])
				self.lexicon1k.append([])            
		else:
			print('unknown dataset!!!')
			exit()
	
	def __getitem__(self, index):
		return  self.pull_item(index)

	def __len__(self):
		return len(self.img_lists)

   

	def pull_item(self, index):
		img_path = self.img_lists[index]
		img = cv2.imread(img_path)
		width, height = self.args.load_width, self.args.load_height
		img = cv2.resize(img.copy(), (width, height))
		img = img[:, :, (2, 1, 0)] ## rgb
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

		gt = self.gts[index]
		lexicon50 = self.lexicon50[index]
		lexicon1k = self.lexicon1k[index]
		lexiconfull = self.full_lexicon
		return torch.from_numpy(img).permute(2, 0, 1).float(), gt, lexicon50, lexicon1k, lexiconfull, img_path


# def samples_collate(batch):
#     imgs = []
#     segs = []
#     for sample in batch:
#         imgs.append(sample[0])
#         segs.append(sample[1])

#     return torch.stack(imgs, 0), torch.stack(segs, 0)

