import os
from PIL import Image


def filter_out_images(root_dir, src, dst):
	lines = open(os.path.join(root_dir, src), 'r').readlines()
	fp = open(os.path.join(root_dir, dst), 'w')

	for line in lines:
		parts = line.strip().split()
		try:
			img = Image.open(os.path.join(root_dir, parts[0]))
			w, h = img.size
			if h > 10 and h < 500 and w > 20 and w < 500:
				fp.write(line)
			else:
				print(parts[0])
		except IOError:
			print "can not open " + parts[0]


root_dir = '/data/data/text_recognition/mnt/ramdisk/max/90kDICT32px/'


filter_out_images(root_dir, 'annotation_train.txt', 'annotation_train_new.txt')
filter_out_images(root_dir, 'annotation_val.txt', 'annotation_val_new.txt')
filter_out_images(root_dir, 'annotation_test.txt', 'annotation_test_new.txt')