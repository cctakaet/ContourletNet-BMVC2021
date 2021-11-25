import os, subprocess, logging
def some_setting():
    powerfulGPU = False
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu_id = "0"
        gpu_list = str(subprocess.check_output(["nvidia-smi", "-L"]))
        all_gpu = ['TITAN RTX', 'GeForce GTX 1080 Ti', 'TITAN Xp', 'GeForce RTX 3090']
        rtxTitan = all_gpu[0] if all_gpu[0] in gpu_list else all_gpu[-1]
        gtx = all_gpu[1] if all_gpu[1] in gpu_list else all_gpu[2]
        choose_gpu = rtxTitan if powerfulGPU else gtx
        for idx, gpu in enumerate(gpu_list.split("\\")):
            if 'GPU %d: ' % idx + choose_gpu in gpu:
                gpu_id = str(idx)
                print('choose GPU: '+choose_gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
some_setting()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse
import time

import utils

import torch
import torchvision.transforms.functional as TF
import cyclegan_networks as cycnet

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='EVAL_DERAIN')
parser.add_argument('--dataset', type=str, default='real', metavar='str',
                    help='dataset name from [rain100h, rain100l, rain800, rain800-real, '
                         'did-mdn-test1, did-mdn-test2, rain1400],'
                         '(default: rain100h)')
parser.add_argument('--in_size', type=int, default=[320, 480], metavar='N', nargs='*', 
                    help='size of input image during eval')
parser.add_argument('--ckpt', type=str, default='ckpt/r100h',
                    help='checkpoints')
parser.add_argument('--net_G', type=str, default='dwt', metavar='str')
parser.add_argument('--nlevels', type=int, default=[4], metavar='N', nargs='*', 
                    help='contourlet dirs (default: 4)')
parser.add_argument('--resize', action='store_true', default=True,
                    help='resize the input images or not')
parser.add_argument('--save_output', action='store_true', default=False,
                    help='to save the output images')
parser.add_argument('--output_dir', type=str, default='eval_output', metavar='str')
parser.add_argument('--real_dir', type=str, default='input_img/moderate', metavar='str')
args = parser.parse_args()




def load_model(args):

    output_nc = 6 if args.mix else 3
    net_G = cycnet.define_G(
                input_nc=3, output_nc=output_nc, ngf=64, netG=args.net_G,
                use_dropout=False, norm='none', nlevels=args.nlevels).to(device)
    total_params = sum(p.numel() for p in net_G.parameters())
    print('loading the best checkpoint...  Total parameters: %d' % total_params)
    # checkpoint = torch.load(os.path.join(args.ckptdir, 'best_ckpt.pt'))
    checkpoint = torch.load(args.ckpt+'.pt')
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    net_G.to(device)
    net_G.eval()

    return net_G

def run_test(args):

    print('Start testing at:', time.ctime())

    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)

    running_time = []

    if args.dataset == 'real':
        datadir = args.real_dir
        val_dirs = glob.glob(os.path.join(datadir, '*.png')) + glob.glob(os.path.join(datadir, '*.jpg'))

    for idx in range(len(val_dirs)):

        this_dir = val_dirs[idx]

        img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        # we recommend to use TF.resize since it was also used during trainig
        # You may also try cv2.resize, but it will produce slightly different results
        if isinstance(args.in_size, (tuple, list)) and len(args.in_size) > 1:
            in_size = args.in_size[:2]
        else:
            in_size = [args.in_size[0], args.in_size[0]]

        if args.resize:
            img_mix = TF.resize(TF.to_pil_image(img_mix), in_size)
        img_mix = TF.to_tensor(img_mix).unsqueeze(0)

        with torch.no_grad():
            tic = time.time()
            img_mix = img_mix.to(device)
            tmp = net_G(img_mix)
            if isinstance(tmp, (list, tuple)):
                tmp = tmp[0]
            toc = time.time()
            G_pred1 = tmp[:, 0:3, :, :]

        G_pred1 = np.array(G_pred1.cpu().detach())
        G_pred1 = G_pred1[0, :].transpose([1, 2, 0])

        G_pred1[G_pred1 > 1] = 1
        G_pred1[G_pred1 < 0] = 0

        running_time.append(toc-tic)

        if args.save_output:
            fname = os.path.basename(this_dir)
            plt.imsave(os.path.join(args.output_dir, fname), G_pred1)

        print('id: %d, running time: %.4f'
              % (idx, np.mean(running_time)), end='\r')

    print('Dataset: %s, running time: %.4f\n'
          % (args.real_dir, np.mean(running_time)))

if __name__ == '__main__':

    test_dict = vars(args)
    import json
    with open(args.ckpt +'.json', 'r') as f:
        test_dict.update(json.load(f))
    args = argparse.Namespace(**test_dict)

    net_G = load_model(args)
    run_test(args)





