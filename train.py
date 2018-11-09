import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from dataloader import SynthLoader, text_collate, SceneLoader
from utils import setup_logger, print_args, strLabelConverter
from model import RAN
import numpy as np
import time
import logging
from warpctc_pytorch import CTCLoss


def main():
	parser = argparse.ArgumentParser(description='RAN')
	parser.add_argument('--name', default='32x100', type=str)
	parser.add_argument('--exp', default='syn90k', type=str)
	
	## data setting 
	parser.add_argument('--root', default='/data/data/text_recognition/',type=str)
	parser.add_argument('--train_dataset', default='synthtext', type=str)
	parser.add_argument('--test_dataset', default='ic03', type=str)
	parser.add_argument('--vis_gt', default=False, type=bool)
	parser.add_argument('--vis_gt_path', default='./outputs/vis_gt', type=str)
	parser.add_argument('--load_width', default=100, type=int)
	parser.add_argument('--load_height', default=32, type=int)

	## model setting
	parser.add_argument('--alphabeta', default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', type=str)
	parser.add_argument('--ignore_case', default=True, type=bool)
	parser.add_argument('--max_len', default=26, type=int)
	## optim setting
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--resume', default='', type=str)
	parser.add_argument('--num_workers', default=8, type=int)
	parser.add_argument('--lr', default=0.0001, type=float)
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
	parser.add_argument('--momentum', default=0.9, type=float)
	parser.add_argument('--weight_decay', default=1e-5, type=float)
	parser.add_argument('--gamma', default=0.1, type=float)
	parser.add_argument('--optim', default='adam', type=str, help='sgd, adam, adadelta')
	# parser.add_argument('--clip_grad', default=False, type=bool)
	parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
	parser.add_argument('--max_epoches', default=1000, type=int)
	# parser.add_argument('--adjust_lr', default='800, 1600', type=str)

	## output setting
	parser.add_argument('--log_freq', default=100, type=int)
	parser.add_argument('--eval_freq', default=10000,type=int )
	parser.add_argument('--snapshot_freq', default=10000, type=int)
	parser.add_argument('--save_folder', default='weights/', type=str)
	parser.add_argument('--eval_vis_num', default=15, type=int)
	parser.add_argument('--max_iter', default=2000000, type=int)
	
	
	args = parser.parse_args()
	if os.path.exists(args.save_folder) == False:
	    os.mkdir(args.save_folder)
	args.save_folder = args.save_folder + args.name + '/'
	if os.path.exists(args.save_folder) == False:
	    os.mkdir(args.save_folder)
	args.save_folder = args.save_folder + args.exp + '/'
	if os.path.exists(args.save_folder) == False:
	    os.mkdir(args.save_folder)
	log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
	##
	if args.ignore_case:
		args.alphabeta = args.alphabeta[:36]
	args.nClasses = len(args.alphabeta) + 1

	setup_logger(log_file_path)
	print_args(args)
	torch.set_default_tensor_type('torch.FloatTensor')

	## setup converter
	converter = strLabelConverter(args.alphabeta, args.ignore_case)

	## setup dataset
	logging.info('model will be trained on %s'%(args.train_dataset))
	trainset =  SynthLoader(args, args.train_dataset, converter)
	logging.info('%d training samples'%(trainset.__len__()))
	train_loader = data.DataLoader(trainset, args.batch_size, num_workers=args.num_workers,
	                              shuffle=True, collate_fn=text_collate, pin_memory=True)

	logging.info('model will be evaluated on %s'%(args.test_dataset))
	testset =  SceneLoader(args, args.test_dataset, False)
	logging.info('%d test samples'%(testset.__len__()))
	test_loader = data.DataLoader(testset, args.batch_size, num_workers=args.num_workers,
	                              shuffle=False,  pin_memory=True)

	## setup model
	net = RAN(args)
	if args.resume:
		logging.info('Resuming training, loading {}...'.format(args.resume))
		net.load_state_dict(torch.load(args.resume))

	net = torch.nn.DataParallel(net).cuda()

	## setup optimizer
	if args.optim == 'sgd':
	    optimizer = optim.SGD(net.parameters(), lr=args.lr,
	                          momentum=args.momentum, weight_decay=args.weight_decay)
	    logging.info('model will be optimed by sgd')
	elif args.optim == 'adam':
	    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	    logging.info('model will be optimed by adam')
	elif args.optim == 'adadelta':
	    optimizer = optim.Adadelta(net.parameters())
	    logging.info('model will be optimed by adadelta')
	else:
	    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	    logging.info('model will be optimed by adam')

	## setup criterion
	criterion = nn.CrossEntropyLoss()

	## train model
	cudnn.benchmark = True
	net.train()
	iter_counter = 0
	acc_max = 0

	for i in range(args.max_epoches):
		t0 = time.time()
		for j, batch_samples in enumerate(train_loader):
			imgs, labels, paths = batch_samples
			imgs = Variable(imgs.float()).cuda()
			labels = Variable(labels.long()).cuda()
			

			preds = net(imgs)
			loss = criterion(preds.view(-1, args.nClasses), labels.view(-1))
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(net.parameters(), args.max_norm)
			optimizer.step()

			if iter_counter % args.log_freq == 0:
				t1 = time.time()
				logging.info('%s->  epoch:%3d  iter:%6d  loss:%4.6f  %4.6fs/batch'%(args.train_dataset, i, j, loss.data[0], (t1-t0)/args.log_freq))
				t0 = time.time()

			if iter_counter % args.snapshot_freq == 0:
				logging.info('Saving state, epoch: %d iter:%d'%(i, j))
				torch.save(net.state_dict(), args.save_folder + '/'  + repr(i) + '_' + repr(j) + '.pth')

			if iter_counter % args.eval_freq == 0:
				## eval model
				net.eval()
				n_correct = 0
				vis_counter = 0
				for index, sample in enumerate(test_loader):
					imgs, gts, lexicon50, lexicon1k, lexiconfull, img_paths = sample
					imgs = Variable(imgs).cuda()
					preds = net(imgs)  ## n*k*c
					preds_size = torch.IntTensor([preds.size(1)] * preds.size(0))
					## decode
					_, preds = preds.max(2)
					preds = preds.contiguous().view(-1)
					text_preds = converter.decode(preds.data, preds_size, raw=False)
					for pred, target in zip(text_preds, gts):
						vis_counter += 1
						if pred == target:
							n_correct += 1
							if vis_counter <= args.eval_vis_num:
								logging.info('pred: %s  gt:%s '%(pred, target))
						else:
							if vis_counter <= args.eval_vis_num:
								logging.info('pred: %s  gt:%s  -----------------------------!!!!!!'%(pred, target))

				acc = n_correct*1.0/testset.__len__()
				if acc > acc_max:
					acc_max = acc
				logging.info('accuracy=%f   acc_max=%f'%(acc, acc_max))

				net.train()

			if iter_counter > args.max_iter:
				break
			iter_counter += 1




	torch.save(net.state_dict(), args.save_folder + '/final_0.pth')
	logging.info('The training stage on %s is over!!!' % (args.train_dataset))


if __name__ == '__main__':
    main()