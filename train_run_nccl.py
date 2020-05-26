import argparse
import time
import random
from torch.utils.data import DataLoader
from core.data.utils import *
from functools import partial
import os
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from core.models.ffn import FFN
from core.data import BatchCreator
from pathlib import Path
import natsort
import adabound
import pickle
import io
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import time

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).')

parser.add_argument('-train_data', '--train_data_dir', type=str, default='/home/xiaotx/2017EXBB/train_data/thick_dense_sparse_coord_d/', help='training data')
parser.add_argument('-b', '--batch_size', type=int, default=8, help='training batch size')
#parser.add_argument('--lr', type=float, default=1e-4, help='training learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='multiplicative factor of learning rate decay')
#parser.add_argument('--step', type=int, default=1e4*2, help='adjust learning rate every step')
parser.add_argument('--depth', type=int, default=12, help='depth of ffn')
parser.add_argument('--delta', default=(15, 15, 15), help='delta offset')
parser.add_argument('--input_size', default=(51, 51, 51), help ='input size')

parser.add_argument('--resume', type=str, default=None, help='resume training')
parser.add_argument('--save_path', type=str, default='/home/xiaotx/2017EXBB/model', help='model save path')
parser.add_argument('--save_interval', type=str, default=1000, help='model save interval')
parser.add_argument('--log_save_path', type=str, default='/home/xiaotx/2017EXBB/model/model_log/', help='model_log save path')





parser.add_argument('--clip_grad_thr', type=float, default=0.7, help='grad clip threshold')
parser.add_argument('--interval', type=int, default=120, help='How often to save model (in seconds).')
parser.add_argument('--iter', type=int, default=1e100, help='training iteration')


parser.add_argument('--stream', type=str, default='nccl_test', help='job_stream')
#launch script need "--local_rank"
parser.add_argument("--local_rank", default=0, type=int)


args = parser.parse_args()

deterministic = args.deterministic
if deterministic:
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def run():

#     cudnn.benchmark = True
#     torch.cuda.set_device(args.local_rank)
    
#     # will read env master_addr master_port world_size
#     torch.distributed.init_process_group(backend='nccl', init_method="env://")
#     args.world_size = dist.get_world_size()
#     args.rank = dist.get_rank()
#     # args.local_rank = int(os.environ.get('LOCALRANK', args.local_rank))
#     args.total_batch_size = (args.batch_size) * dist.get_world_size()
    def dist_init(host_addr, rank, local_rank, world_size, port=23456):
        host_addr_full = 'tcp://' + host_addr + ':' + str(port)
        torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()

    args.rank = int(os.environ['SLURM_PROCID'])
    args.local_rank = int(os.environ['SLURM_LOCALID'])
    args.world_size = int(os.environ['SLURM_NTASKS'])
    args.ip = get_ip(os.environ['SLURM_STEP_NODELIST'])
    dist_init(args.ip, args.rank, args.local_rank, args.world_size)
    global resume_iter
    """model_log"""
    input_size_r = list(args.input_size)
    delta_r = list(args.delta)

    path = args.log_save_path + "model_log_fov:{}_delta:{}_depth:{}".format(input_size_r [0],delta_r[0],args.depth)
    filesize = os.path.getsize(path)
    if filesize == 0:

        f = open(path, 'wb')
        data_start = {'chris': "xtx"}
        pickle.dump(data_start, f)
        f.close()
    else:
        f = open(path, 'rb')
        data = pickle.load(f)
        resume_iter = len(data.keys())-1
        f.close()


    """model_construction"""
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)


    """data_load"""
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))


    abs_path_training_data = args.train_data_dir
    entries_train_data = Path(abs_path_training_data )
    files_train_data = []

    for entry in entries_train_data.iterdir():
        files_train_data.append(entry.name)

    sorted_files_train_data = natsort.natsorted(files_train_data, reverse=False)

    files_total = len(sorted_files_train_data)

    input_h5data_dict = {}
    train_dataset_dict = {}
    train_loader_dict = {}
    batch_it_dict = {}
    train_sampler_dict = {}

    for index in range(files_total):
        input_h5data_dict[index] = [(abs_path_training_data + sorted_files_train_data[index])]
        print(input_h5data_dict[index])
        train_dataset_dict[index] = BatchCreator(input_h5data_dict[index], args.input_size, delta=args.delta, train=True)
        train_sampler_dict[index] = torch.utils.data.distributed.DistributedSampler(train_dataset_dict[index], num_replicas=args.world_size, rank=args.rank, shuffle=True)
        train_loader_dict[index] = DataLoader(train_dataset_dict[index], num_workers=0, sampler=train_sampler_dict[index] , pin_memory=True)
        batch_it_dict[index] = get_batch(train_loader_dict[index], args.batch_size, args.input_size,
                               partial(fixed_offsets, fov_moves=train_dataset_dict[index].shifts))




    best_loss = np.inf

    """optimizer"""
    t_last = time.time()
    cnt = 0
    tp = fp = tn = fn = 0
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3) 
    #momentum=0.9 
    #optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=args.gamma, last_epoch=-1)


    """train_loop"""
    while cnt < args.iter:
        cnt += 1

        Num_of_train_data = len(input_h5data_dict)
        index_rand = random.randrange(0, Num_of_train_data, 1)

        seeds, images, labels, offsets = next(batch_it_dict[index_rand])
        #print(sorted_files_train_data[index_rand])
        #seeds = seeds.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        #offsets = offsets.cuda(non_blocking=True)

        t_curr = time.time()

        labels = labels.cuda(non_blocking=True)

        torch_seed = torch.from_numpy(seeds).cuda(non_blocking=True)
        input_data = torch.cat([images, torch_seed], dim=1)
        input_data = Variable(input_data.cuda(non_blocking=True))

        logits = model(input_data)
        updated = torch_seed + logits

        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(updated, labels)
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_thr)
        optimizer.step()
        
        
        seeds[...] = updated.detach().cpu().numpy()

        pred_mask = (updated >= logit(0.9)).detach().cpu().numpy()
        true_mask = (labels > 0.5).cpu().numpy()
        true_bg = np.logical_not(true_mask)
        pred_bg = np.logical_not(pred_mask)
        tp += (true_mask & pred_mask).sum()
        fp += (true_bg & pred_mask).sum()
        fn += (true_mask & pred_bg).sum()
        tn += (true_bg & pred_bg).sum()
        precision = 1.0 * tp / max(tp + fp, 1)
        recall = 1.0 * tp / max(tp + fn, 1)
        accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
        if args.rank == 0:
            print('[Iter_{}:, loss: {:.4}, Precision: {:.2f}%, Recall: {:.2f}%, Accuracy: {:.2f}%]\r'.format(
            cnt, loss.item(), precision*100, recall*100, accuracy * 100))

        #scheduler.step()


        """model_saving_(best_loss)"""
        """
        if best_loss > loss.item() or t_curr - t_last > args.interval:
            tp = fp = tn = fn = 0
            t_last = t_curr
            best_loss = loss.item()
            input_size_r = list(args.input_size)
            delta_r = list(args.delta)
            torch.save(model.state_dict(), os.path.join(args.save_path,
                                                        'ffn_model_fov:{}_delta:{}_depth:{}.pth'.format(input_size_r[0],
                                                                                                        delta_r[0],
                                                                                                        args.depth)))
            print('Precision: {:.2f}%, Recall: {:.2f}%, Accuracy: {:.2f}%, Model saved!'.format(
                precision * 100, recall * 100, accuracy * 100))
        """

        """model_saving_(iter)"""


        if (cnt % args.save_interval) == 0 and args.rank == 0:
            tp = fp = tn = fn = 0
            #t_last = t_curr
            #best_loss = loss.item()
            input_size_r = list(args.input_size)
            delta_r = list(args.delta)
            torch.save(model.state_dict(), os.path.join(args.save_path, (str(args.stream) + 'ffn_model_fov:{}_delta:{}_depth:{}_recall{}.pth'.format(input_size_r [0],delta_r[0],args.depth,recall*100))))
            print('Precision: {:.2f}%, Recall: {:.2f}%, Accuracy: {:.2f}%, Model saved!'.format(
                precision * 100, recall * 100, accuracy * 100))


            path = args.log_save_path + "model_log_fov:{}_delta:{}_depth:{}".format(input_size_r [0],delta_r[0],args.depth)
            model_eval = "precision#" + str('%.4f' % (precision * 100)) + "#recall#" + str('%.4f' % (recall * 100)) + "#accuracy#" + str('%.4f' % (accuracy * 100))

            f_l = open(path, 'rb')
            data = pickle.load(f_l)

            key =  cnt/args.save_interval + resume_iter
            data[key] = model_eval

            f_o = open(path, 'wb')
            pickle.dump(data, f_o)

            f_o.close()
            f_l.close()


if __name__ == "__main__":
    seed = int(time.time())
    random.seed(seed)
    time1 = time.time()
    run()
    print("run time:", time.time() - time1)
