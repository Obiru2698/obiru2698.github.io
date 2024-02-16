import os
import argparse
from glob import glob
from model import A4RNet
import torch
import time


parser = argparse.ArgumentParser(description='')
device = torch.device('cuda:0')
parser.add_argument('--gpu_id', dest='gpu_id', 
                    default="0",
                    help='GPU ID (-1 for CPU)')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', 
                    default='./ckpts/',
                    help='directory for checkpoints')

####test####

parser.add_argument('--data_dir', dest='data_dir',
                    default='./data/test/phone/',
                    help='directory storing the test data')
parser.add_argument('--res_dir', dest='res_dir',
                    default='./results/test/phone/',
                    help='directory for saving the results')

###compare###
# parser.add_argument('--data_dir', dest='data_dir',
#                     default='./data/darkface/',
#                     help='directory storing the test data')
# parser.add_argument('--res_dir', dest='res_dir', 
#                     default='./results/test/other_model_v5/',
#                     help='directory for saving the results')

####eval####
# parser.add_argument('--data_dir', dest='data_dir',
#                     default='./Eval/low/',
#                     help='directory storing the test data')




args = parser.parse_args()


def predict(model):

    test_low_data_names  = glob(args.data_dir + '/' + '*.*')
    test_low_data_names.sort()
    print('Number of evaluation images: %d' % len(test_low_data_names))

    model.predict(test_low_data_names,
                res_dir=args.res_dir,
                ckpt_dir=args.ckpt_dir)



if __name__ == '__main__':
    if args.gpu_id != "-1":
        # Create directories for saving the results
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Create the model
        with torch.no_grad():
            model = A4RNet().to(device)
            # Test the model
            predict(model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError