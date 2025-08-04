import argparse
import errno
import tensorboardX
from tqdm import tqdm
from time import time
import prettytable
import copy
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.utils.logging import Logger
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.data.datareader_h36m import DataReaderH36M  
from lib.model.loss import *
from datetime import datetime


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/MB_train_h36m.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)
    
def evaluate(args, model_pos, test_loader, datareader):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()            
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.rootrel:
                predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,0,2] = 0

            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()

    actions = np.array(datareader.dt_dataset['test']['action'][::args.sample_stride])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'][::args.sample_stride])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'][::args.sample_stride])
    sources = np.array(datareader.dt_dataset['test']['source'][::args.sample_stride])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    assert len(results_all)==len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    results_joints = [{} for _ in range(args.num_joints)]
    results_accelaration = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_accelaration[action] = []
        for joint_idx in range(args.num_joints):
            results_joints[joint_idx][action] = []

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:,None,None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:,0:1,:]
        gt = gt - gt[:,0:1,:]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        acc_err = mpjve(pred, gt)
        acc_err_all[frame_list[:-1]] += acc_err
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            acc_err = acc_err_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
            results_accelaration[action].append(acc_err)
    final_result = []
    final_result_procrustes = []
    final_result_acceleration = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        final_result_acceleration.append(np.mean(results_accelaration[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    summary_table.add_row(['MPJVE'] + final_result_acceleration)
    print(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    acceleration_error = np.mean(np.array(final_result_acceleration))
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('MPJVE:', acceleration_error, 'mm/s')
    print('----------')
    return e1, e2, results_all


def train_with_config(args, opts):
    TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
    file_path = 'log/default'+'_'+TIMESTAMP
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    logfile = os.path.join(file_path, 'logging.log')
    sys.stdout = Logger(logfile)

    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)


    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)


    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)

    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            #model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            state_dict = checkpoint['model_pos']
            if 'module.joints_embed.weight' in state_dict:
                if state_dict['module.joints_embed.weight'].shape != model_backbone.state_dict()[
                    'module.joints_embed.weight'].shape:
                    print("Removing mismatched key: module.joints_embed.weight")
                    del state_dict['module.joints_embed.weight']
            model_backbone.load_state_dict(state_dict, strict=False)
            model_pos = model_backbone            
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone

                
    if opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)