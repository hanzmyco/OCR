import os
import torch
from torch.autograd import Variable
import torch.utils.data as data
from dataloader import SynthLoader, text_collate, SceneLoader
from utils import setup_logger, print_args, strLabelConverter
from model import CRNN
import argparse
import numpy as np
import time
import logging



def main():
	parser = argparse.ArgumentParser(description='CRNN')
	parser.add_argument('--name', default='32x100', type=str)
	parser.add_argument('--exp', default='syn90k', type=str)
	
	## data setting 
	parser.add_argument('--root', default='/data/data/text_recognition/',type=str)
	parser.add_argument('--test_dataset', default='ic03', type=str)
	parser.add_argument('--load_width', default=100, type=int)
	parser.add_argument('--load_height', default=32, type=int)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--num_workers', default=8, type=int)
	## model setting
	parser.add_argument('--snapshot', default='./weights/32x100/syn90k/3_51474.pth', type=str)
	parser.add_argument('--alphabeta', default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', type=str)
	parser.add_argument('--ignore_case', default=True, type=bool)
	## output setting
	parser.add_argument('--out_dir', default='./outputs', type=str)

	args = parser.parse_args()

	if os.path.exists(args.out_dir) == False:
		os.mkdir(args.out_dir)
	args.out_dir = os.path.join(args.out_dir, args.name)
	if os.path.exists(args.out_dir) == False:
		os.mkdir(args.out_dir)
	args.out_dir = os.path.join(args.out_dir, args.snapshot.strip().split('/')[-1].split('.')[0])
	if os.path.exists(args.out_dir) == False:
		os.mkdir(args.out_dir)

	if args.ignore_case:
		args.alphabeta = args.alphabeta[:36]
	args.nClasses = len(args.alphabeta) + 1

	log_path = os.path.join(args.out_dir, args.test_dataset + '.txt')
	setup_logger(log_path)

	logging.info('model will be evaluated on %s'%(args.test_dataset))
	testset =  SceneLoader(args, args.test_dataset, False)
	logging.info('%d test samples'%(testset.__len__()))
	test_loader = data.DataLoader(testset, args.batch_size, num_workers=args.num_workers,
	                              shuffle=False,  pin_memory=True)

	## model
	net = CRNN(args)
	print(net)
	input()
	net = torch.nn.DataParallel(net).cuda()
	print(net)
	net.load_state_dict(torch.load(args.snapshot))
	net = net.module
	net.eval()
	n_correct = 0
	n_samples = 0
	converter = strLabelConverter(args.alphabeta, args.ignore_case)

	for index, sample in enumerate(test_loader):
		# print('model state', net.training)
		# print('bn1 state', net.cnn[0].training)
		# print('conv1.weight', net.cnn[0].subnet[0].weight[0, 0, 0, 0])
		# print('bn1.weight', net.cnn[4].subnet[1].weight[0])
		# print('bn1.bias', net.cnn[4].subnet[1].weight[0])
		# print('bn1.running_mean', net.cnn[4].subnet[1].running_mean[0])
		# print('bn1.running_var', net.cnn[4].subnet[1].running_var[0])
		imgs, gts, lexicon50, lexicon1k, lexiconfull, img_paths = sample
		imgs = Variable(imgs).cuda()
		preds = net(imgs)
		preds_size = torch.IntTensor([preds.size(0)] * preds.size(1))
		## decode
		_, preds = preds.max(2)
		preds = preds.transpose(1, 0).contiguous().view(-1)
		text_preds = converter.decode(preds.data, preds_size, raw=False)
		for pred, target in zip(text_preds, gts):
			n_samples += 1
			if pred.lower() == target.lower():
				n_correct += 1
				logging.info('pred: %s  gt:%s '%(pred, target))
			else:
				logging.info('pred: %s  gt:%s  -----------------------------!!!!!!'%(pred, target))
	assert(n_samples == testset.__len__())
	acc = n_correct*1.0/testset.__len__()
	logging.info('accuracy=%f'%(acc))









if __name__ == '__main__':
    main()