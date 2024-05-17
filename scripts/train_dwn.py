import numpy as np
import json
import argparse
import copy
import os
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import average_precision_score
from mmcv.parallel import MMDistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from mmcv.runner import get_dist_info
import torch.distributed as dist

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def train(args, model, train_loader, optimizer, epoch):
    
    model.train()         
    ave_loss=0

    for batch_idx, batch in enumerate(train_loader):              
        optimizer.zero_grad() 
        output = model(batch['data_in'])
        # 計算alpha=0 or 1
        loss = cal_loss(output, batch['gt'])                  
        ave_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch['data_in']), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), flush=True)
        
    ave_loss/=len(train_loader)
    print('\nTraining set: Average loss: {:.4f}\n'.format(ave_loss))    
    return ave_loss

def test(model, test_loader):
    model.eval()
    test_loss = 0
    
    mean_metrics = dict(ap = 0,
                       error_cam = 0,
                       error_rd = 0,
                       error_best = 0,
                       error_prd = 0)
       
    with torch.no_grad():  
        for batch in tqdm(test_loader, 'Validation'): 
            output = model(batch['data_in'])
            loss = cal_loss(output, batch['gt']) 
            metrics = cal_metrics(output, batch['gt'])
            
            test_loss += loss
            for k in mean_metrics:
                mean_metrics[k] += metrics[k]                      
               
    test_loss/= len(test_loader)
    for k in mean_metrics:
        mean_metrics[k] /= len(test_loader)
        print(k, mean_metrics[k])
            
    print('\nValdation set: Average loss: {:.4f}\n'.format(test_loss))
    
    return test_loss

class FusionDataset(Dataset):
    def __init__(self, path_list):
        data = []
        
        for path in path_list:
            data_dict = json.load(open(path, 'r'))
            data_one = np.array(data_dict['data'])
            data.append(data_one)
        
        self.data = np.concatenate(data, axis=0)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        v = self.data[index,:]
        
        d_gt, cls_idx, lvl_cam, score_cam, d_cam, x_cam, y_cam, vx_cam, vz_cam, obj_idx_cam,\
        lvl_rd, score_rd, d_rd, x_rd, y_rd, vx_rd, vz_rd, obj_idx_rd, rcs =  v[1:]
        
        v_prj = vx_cam * vx_rd + vz_cam * vz_rd
        v_mag_rd = vx_rd ** 2 + vz_rd ** 2
        diff_pixel = ( (x_cam - x_rd)**2 + (y_cam - y_rd)**2 ) ** 0.5
        diff_d = abs(d_cam - d_rd)
        
        data_in = torch.from_numpy(np.array([cls_idx, lvl_cam, lvl_rd, score_cam, score_rd,\
                                            d_cam, d_rd, diff_d, vx_cam, vz_cam, vx_rd, vz_rd,\
                                            v_prj, v_mag_rd, diff_pixel, rcs], dtype=np.float32))
            
        gt = torch.from_numpy(np.array([d_gt, d_cam, d_rd, obj_idx_cam, obj_idx_rd], dtype=np.float32))
        
        sample = dict(data_in=data_in, gt=gt)
        
        return sample
    # INFO 定義DWN模型架構

class FusionMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# INFO 定義DWN+BatchNorm1d+L2正規化+dropout
# class FusionMLP(nn.Module):
#     def __init__(self, l2_reg=0.001):
#         super().__init__()
#         self.fc1 = nn.Linear(16, 128)
#         self.bn1 = nn.BatchNorm1d(128)  
#         self.fc2 = nn.Linear(128, 128)
#         self.bn2 = nn.BatchNorm1d(128)  
#         self.fc3 = nn.Linear(128, 64)
#         self.bn3 = nn.BatchNorm1d(64)  
#         self.fc4 = nn.Linear(64, 32)
#         self.bn4 = nn.BatchNorm1d(32)  
#         self.fc5 = nn.Linear(32, 1)
#         self.l2_reg = l2_reg

#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))  
#         x = F.relu(self.bn2(self.fc2(x))) 
#         x = F.relu(self.bn3(self.fc3(x))) 
#         x = F.relu(self.bn4(self.fc4(x))) 
#         x = self.fc5(x)
#         return x

def cal_loss(output, gt):
    
    gt = gt.to(output.device)
    
    d_gt = gt[:,[0]]
    d_cam = gt[:,[1]]
    d_rd = gt[:,[2]]
    
    error_cam = torch.abs(d_cam - d_gt)
    error_rd = torch.abs(d_rd - d_gt)
    
    label = (error_rd < error_cam).to(torch.float32)
    
    n_pos = torch.sum(label)
    n_neg = len(label) - n_pos
    pos_weight = n_neg/n_pos
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)   
    loss = criterion(output, label)
    return loss

def cal_metrics(output, gt):
    gt = gt.to(output.device)
    
    d_gt = gt[:,0]
    d_cam = gt[:,1]
    d_rd = gt[:,2]
    
    error_cam = torch.abs(d_cam - d_gt)
    error_rd = torch.abs(d_rd - d_gt)
    
    label = (error_rd < error_cam).to(torch.float32)
    
    score = output.squeeze(1).detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    
    ap = average_precision_score(label, score)
    
    loss_mae = torch.nn.L1Loss(reduction='mean')
    
    prd_score = output.squeeze(1).sigmoid()
    thres_score = 0.4
    d_cat = torch.stack([d_cam,d_rd], dim=1)
    idx2 = (prd_score > thres_score).to(torch.int64)
    idx1 = torch.arange(d_cat.shape[0], device=d_cat.device)
    
    gt_idx = (error_rd < error_cam).to(torch.int64)
    
    d_prd = d_cat[idx1,idx2]
    d_best = d_cat[idx1,gt_idx]
    
    error_cam = loss_mae(d_rd, d_gt)
    error_rd = loss_mae(d_cam, d_gt)
    error_prd = loss_mae(d_prd, d_gt)
    error_best = loss_mae(d_best, d_gt)
    
    return dict(ap = ap,
                error_cam = error_cam.item(),
                error_rd = error_rd.item(),
                error_best = error_best.item(),
                error_prd = error_prd.item())

def init_env():
    ddp = True
    if ddp:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = None

    device = torch.device("cuda", local_rank) if ddp else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = torch.cuda.is_available()

    available_gpu_ids = list(range(torch.cuda.device_count()))
    return local_rank, device, available_gpu_ids
    
def init_params(args, model, optimizer):
    loss_train=[]
    loss_val=[]
    
    start_epoch = 0
    state_dict_best = None
    loss_val_min = None
        
    if args.resume == True:
        f_checkpoint = join(args.dir_result, 'checkpoint_dwn.tar')        
        if os.path.isfile(f_checkpoint):
            print('Resume training')
            checkpoint = torch.load(f_checkpoint)   
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict) 
            optimizer.load_state_dict(checkpoint['optimizer']) 
            start_epoch = checkpoint['epoch'] + 1
                
            loss_train, loss_val = checkpoint['loss']
            loss_val_min = checkpoint['loss_val_min']
            state_dict_best = checkpoint['state_dict_best']
        else:            
            print('No checkpoint file is found.')
    return loss_train, loss_val, start_epoch, state_dict_best, loss_val_min

def mkdir(dir1):
    if not os.path.exists(dir1): 
        os.makedirs(dir1)
        print('make directory %s' % dir1)

def plot_and_save_loss_curve(args, epoch, loss_train, loss_val):
    plt.close('all')
    plt.figure()
    t = np.arange(epoch + 1)

    # Convert tensors to numpy arrays, ensuring they are on CPU first
    loss_train_cpu = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in loss_train]
    loss_val_cpu = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in loss_val]

    plt.plot(t, loss_train_cpu, 'b.-')
    plt.plot(t, loss_val_cpu, 'r.-')

    plt.grid()
    plt.legend(['Training Loss', 'Validation Loss'], loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')

    # Save the figure. Ensure the directory exists or handle exceptions appropriately
    plt.savefig(join(args.dir_result, 'loss_dwn.png'))


def save_checkpoint(epoch, model, optimizer, loss_train, loss_val, loss_val_min, state_dict_best, args):
    if epoch == 0 or loss_val[-1] < loss_val_min:
        print("into here")
        loss_val_min = loss_val[-1]
        state_dict_best = copy.deepcopy( model.state_dict() )
      
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss': [loss_train, loss_val],
             'loss_val_min': loss_val_min,
             'state_dict_best': state_dict_best}
    
    torch.save(state, join(args.dir_result, 'checkpoint_dwn.tar'))
    return loss_val_min, state_dict_best

def main(args):
    torch.distributed.init_process_group(backend="nccl") # 初始化进程组
    dist.barrier() #等待所有GPU运行，同步所有进程

    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        # args.dir_data = join(this_dir, '..', 'data', 'nuscenes', 'fusion_data', 'dwn_radiant_pgd')
        args.dir_data = "/workspace/camera_radar_swin_trainval/dir_results/lwa/dwn_data"
    
    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result')              
    mkdir(args.dir_result) 
    
    train_ann_files = [join(args.dir_data, 'train.json')] 
    val_ann_files = [join(args.dir_data, 'val.json')]
    # train_ann_files = [join(args.dir_data, 'train_mini.json')] 
    # val_ann_files = [join(args.dir_data, 'val_mini.json')]
    
    local_rank, device, available_gpu_ids = init_env()
    args.num_gpus = len(available_gpu_ids)
    print('%d GPUs' % args.num_gpus)
    
    rank, world_size = get_dist_info()   

    print("===================== setting Fusion Dataset =====================")
    train_dataset = FusionDataset(train_ann_files)
    val_dataset = FusionDataset(val_ann_files)

    print("===================== setting DistributedSampler =====================")
    train_sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=True) 
    val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=True) 
 

    print("===================== setting train dataloader =====================")
    train_loader = DataLoader(dataset = train_dataset,
                        batch_size = args.num_gpus * args.samples_per_gpu,
                        shuffle = False,
                        num_workers = args.num_gpus * args.workers_per_gpu,
                        sampler=train_sampler)
    print("===================== setting val dataloader =====================")
    val_loader = DataLoader(dataset = val_dataset,
                            batch_size = args.num_gpus * args.samples_per_gpu,
                            shuffle = False,
                            num_workers = args.num_gpus * args.workers_per_gpu,
                            sampler=val_sampler)
    
    distributed = True
    if distributed:
        model = FusionMLP()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = MMDistributedDataParallel(model, device_ids=[
                                          local_rank], output_device=local_rank, find_unused_parameters=True)

    else:
        model = FusionMLP()
        model = torch.nn.DataParallel(model.to(device), device_ids=available_gpu_ids)

        
    if not args.do_eval:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        loss_train, loss_val, start_epoch, state_dict_best, loss_val_min = \
            init_params(args, model, optimizer)
          
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(args.epochs)
            val_sampler.set_epoch(args.epochs) 

            start = timer() 
            
            loss_train.append(train(args, model, train_loader, optimizer, epoch))                  
    
            end = timer(); t = (end - start) / 60; print('Training Time used: %.1f minutes\n' % t)
            
            loss_val.append(test(model, val_loader))
            
            plot_and_save_loss_curve(args, epoch, loss_train, loss_val)
            
            loss_val_min, state_dict_best = save_checkpoint(epoch, model, optimizer, loss_train, loss_val, loss_val_min, state_dict_best, args)
                         
            end = timer(); t = (end - start) / 60; print('Time used: %.1f minutes\n' % t)
            
            print('Training Epoch %d finished.' % epoch )
                 
    if args.do_eval:     
        f_checkpoint = join(args.dir_result, 'checkpoint.tar')        
        if os.path.isfile(f_checkpoint):
            print('load model')
            checkpoint = torch.load(f_checkpoint)                    
            model.load_state_dict(checkpoint['state_dict'])
        else:            
           print('No checkpoint file is found.')        
        test(model, val_loader)
    
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dir_data', type=str, default="dir_results/server/ori_server/dwn_data")
    parser.add_argument("--dir_result", type=str, default="dir_results/server/ori_server/dwn_data")
    parser.add_argument('--epochs', type=int, default=200) 
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from checkpoint')    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--samples_per_gpu', type=int, default=256)  
    parser.add_argument('--workers_per_gpu', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=100)   
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument("--local_rank", default=1, type=int)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()

    main(args)

# run multi gpu command 
# python -m torch.distributed.launch --nproc_per_node=4 scripts/train_dwn.py 