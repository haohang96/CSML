import numpy as np
import torch
import torch.nn as nn
import moco.builder
import pdb

from easydict import EasyDict
from absl import flags
from absl import app 

FLAGS = flags.FLAGS


def compute_feat(model, loader, gpu_rank):
    num_feat = 0
    model.eval()
    all_feats = np.zeros([FLAGS.dataset_len+1000, 2048]).astype(np.float32)
    all_index = np.zeros([FLAGS.dataset_len+1000]).astype(np.int)
    
    for i, (images, target, index) in enumerate(loader):
        images = images.cuda(gpu_rank, non_blocking=True)
        index = index.cuda(gpu_rank, non_blocking=True)
        with torch.no_grad():
            k = model.module.encoder_k(images, pre_out=True)
            k = nn.functional.normalize(k, dim=1)

        k = moco.builder.concat_all_gather(k)
        k = k.cpu().numpy()

        index = moco.builder.concat_all_gather(index)
        index = index.cpu().numpy()

        if i < len(loader) - 1:
            bsz = k.shape[0]
            all_feats[i*bsz: (i+1)*bsz] = k
            all_index[i*bsz: (i+1)*bsz] = index
            num_feat += bsz
        else:
            all_feats[i*bsz: i*bsz + k.shape[0]] = k
            all_index[i*bsz: i*bsz + k.shape[0]] = index
            num_feat += k.shape[0]

        if i%200 == 0:
            print('%d | %d'%(i, len(loader)))

    print('num_feat: ', num_feat)
    model.train()
    all_feats = all_feats[:num_feat]
    all_index = all_index[:num_feat]

    sorted_index, sort_id = np.unique(all_index, return_index=True)
    sorted_feats = all_feats[sort_id]
    assert (all_index[sort_id] == np.arange(0, FLAGS.dataset_len)).all()

    return sorted_feats


def knn(feat):
    if FLAGS.moxing:
        import faiss
    else:
        import mkl
        mkl.get_max_threads()
        import faiss
    d = feat.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    # index = faiss.index_cpu_to_all_gpus(cpu_index)
    index = cpu_index # only for debug
    index.add(feat)

    D, I = index.search(feat, FLAGS.clus_pos_num + 1) # self-image is include in I[:,0]
    imgs_corr = [[] for i in range(FLAGS.dataset_len)]
    for i in range(FLAGS.dataset_len):
        for j in range(FLAGS.clus_pos_num):
            imgs_corr[i].append(I[i,j+1])

    imgs_corr = np.array(imgs_corr) # 1281167*FLAGS.clus_pos_num ndarray
    return EasyDict(imgs_corr=imgs_corr)
