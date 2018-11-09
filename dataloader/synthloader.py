import os
import os.path as op
import sys
import cv2
import random
from PIL import Image, ImageDraw
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from scipy.misc import imread, imresize
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SynthLoader(data.Dataset):
	def __init__(self, args, dataset, converter, is_training=True, aug=False):
		self.root = args.root
		self.dataset = dataset
		self.args = args
		self.converter = converter
		self.is_training = is_training
		self.aug = aug
		self.get_all_samples()

	def parse_samples(self, img_list, lexicon):
		res_imgs = []
		res_labels = []
		for line in img_list:
			parts = line.strip().split()
			res_imgs.append(parts[0])
			res_labels.append(lexicon[int(parts[-1])])
		return res_imgs, res_labels




	def get_all_samples(self):
		## check datasets
		assert (self.dataset == 'synthtext')
		self.lexicon = [x.strip() for x in open(op.join(self.root, 'mnt/ramdisk/max/90kDICT32px', 'lexicon.txt')).readlines()]
		if self.is_training:
			self.train_list = open(op.join(self.root, 'mnt/ramdisk/max/90kDICT32px', 'annotation_train_new.txt')).readlines()
			self.train_imgs, self.train_labels = self.parse_samples(self.train_list, self.lexicon)
			if self.aug:
				self.val_list = open(op.join(self.root, 'mnt/ramdisk/max/90kDICT32px', 'annotation_val_new.txt')).readlines()
				self.val_imgs, self.val_labels = self.parse_samples(self.val_list, self.lexicon)
				self.image_paths = self.train_imgs + self.val_imgs
				self.image_labels = self.train_labels + self.val_labels
			else:
				self.image_paths = self.train_imgs
				self.image_labels = self.train_labels
		else:
			self.test_list = open(op.join(self.root, 'mnt/ramdisk/max/90kDICT32px', 'annotation_test_new.txt')).readlines()
			self.image_paths, selg.image_labels = self.parse_samples(self.test_list, self.lexicon)
	
	def vis_gt(self, image, label, save_path):
		if os.path.exists(save_path) == False:
			os.mkdir(save_path)
		im = image.numpy().astype(np.uint8).transpose((1, 2, 0))
		save_name = os.path.join(save_path, label + '.jpg')
		Image.fromarray(im).save(save_name)


			

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		image_path = op.join(self.root,  'mnt/ramdisk/max/90kDICT32px', self.image_paths[index])
		image_label = self.image_labels[index]
		# img = imread(image_path, mode='RGB')
		img = Image.open(image_path)
		width, height = self.args.load_width, self.args.load_height
		img_resized = img.resize((width, height))
		img_resized = np.array(img_resized).astype(np.float32)
		img_resized = torch.from_numpy(img_resized.transpose((2, 0, 1)))
		if self.args.vis_gt:
			self.vis_gt(img_resized, image_label, self.args.vis_gt_path)
		text, length = self.converter.encode(image_label)

		text = torch.IntTensor((text + [0] * self.args.max_len)[:self.args.max_len])

		return img_resized, text, image_path


def text_collate(batch):
	imgs = []
	labels = []
	paths = []
	for sample in batch:
		imgs.append(sample[0])
		labels.append(sample[1])
		paths.append(sample[2])
	imgs = torch.stack(imgs, 0)
	labels = torch.cat(labels, 0)
	return imgs, labels, paths