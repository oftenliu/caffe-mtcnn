import argparse
import os,sys
import numpy as np
import numpy.random as npr
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)




# keep sample ratio [neg, pos, part] = [3, 1, 1]

def start(net,shuffling=False):

	saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))

	lFileName = os.path.join(saveFolder, "train_%s.txt"%(net))

	with open(os.path.join(saveFolder, 'pos.txt'), 'r') as f:
		pos = f.readlines()
	with open(os.path.join(saveFolder, 'neg.txt'), 'r') as f:
		neg = f.readlines()
	with open(os.path.join(saveFolder, 'part.txt'), 'r') as f:
		part = f.readlines()

	base_num = min([len(neg), len(pos), len(part)])
	if len(neg) > base_num * 3:
		neg_keep = np.random.choice(len(neg), size=base_num * 3, replace=False)
	else:
		neg_keep = np.random.choice(len(neg), size=len(neg), replace=False)
	pos_keep = np.random.choice(len(pos), size=base_num, replace=False)
	part_keep = np.random.choice(len(part), size=base_num, replace=False)

	dataset = []
	for i in pos_keep:
		dataset.append(pos[i])
	for i in neg_keep:
	    dataset.append(neg[i])
	for i in part_keep:
		dataset.append(part[i])


	if shuffling:
		np.random.shuffle(dataset)

	with open(lFileName, "w") as f:
		for i, image_example in enumerate(dataset):
			f.writelines(image_example)

			print('\rConverting[%s]: %d/%d' % (net, i + 1, len(dataset)))

	print('\nFinished converting the MTCNN dataset!')


def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='unknow', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
	stage = 'pnet'
	args = parse_args()
	if stage not in ['pnet', 'rnet', 'onet']:
		raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # set GPU
	if args.gpus:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
	start(stage, True)


