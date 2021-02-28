import torch
import torch.nn as nn

import numpy as np
import os
import imageio
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths

def validate(fid_stat, gen_net: nn.Module, G_input_dim):
    # eval mode
    gen_net = gen_net.eval()


    # get fid and inception score
    fid_buffer_dir = 'fid_buffer'
    os.makedirs(fid_buffer_dir)

    eval_iter = 400
    img_list = list()
    for iter_idx in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, G_input_dim)))
        z = z.view(-1, G_input_dim, 1, 1)

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, 'iter%d_b%d.png' % (iter_idx, img_idx))
            imageio.imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    mean, std = get_inception_score(img_list)

    # get fid score
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    os.system('rm -r {}'.format(fid_buffer_dir))

    return mean, fid_score