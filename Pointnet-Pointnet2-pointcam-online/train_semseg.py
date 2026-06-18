# ---------------------------------------------------------------
# Modified from 'Pointnet_Pointnet2_pytorch ' 
# Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch 
# ---------------------------------------------------------------
import sys
sys.dont_write_bytecode = True 
import os

import argparse
import torch
import torch.nn as nn
import datetime
import logging
import importlib
import shutil
import provider
import numpy as np
import time
import scipy.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import anom_utils
import data_utils.S3DISDataLoader as S3DISDataLoader
from data_utils.S3DISDataLoader import S3DISDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


#######################################################
def eval_open_measure(conf, seg_label, args, mask=None):
    out_labels = args.out_labels
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]
    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores  = - conf[np.logical_not(out_label)]
    out_scores = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any open pixels.")
        return 0.0, 0.0, 0.0


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
#######################################################


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=64, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    ################################################################
    parser.add_argument('--data_split', type=str, default='s3dis_1')  # [s3dis_1, s3dis_3]
    parser.add_argument('--open_eval',  type=str, default='pointcam') # [msp, maxlogit, pointcam]
    parser.add_argument('--out_labels', default=(13, ), type=int)
    parser.add_argument('--use_cutmix',  action='store_true')
    parser.add_argument('--use_pointcam',  action='store_true')

    parser.add_argument('--alpha',  default=0.0,  type=float)
    parser.add_argument('--select_ratio', default=0.0, type=float)
    ################################################################
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    #################################################
    if args.data_split == 's3dis_1':
        classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'bookcase',
                   'board', 'clutter', 'unknown']
    elif args.data_split == 's3dis_3':
        classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'bookcase',
                   'board', 'clutter', 'unknown']
    else: 
        classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
                   'board', 'clutter']

    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat

    # root = '/scratch1/hon045/data/s3dis/stanford_indoor3d/'
    root = '/OSM/CBR/D61_RCV/scratch/hon045/s3dis/stanford_indoor3d/'

    S3DISDataLoader.select_ratio = args.select_ratio
    if args.data_split == 's3dis_1':
        NUM_CLASSES = 12
        if args.use_cutmix:
            NUM_CLASSES = 12+1
            S3DISDataLoader.NEW_LABEL = NUM_CLASSES-1

    elif args.data_split == 's3dis_3':
        NUM_CLASSES = 10
        if args.use_cutmix:  
            NUM_CLASSES = 10+1
            S3DISDataLoader.NEW_LABEL = NUM_CLASSES-1
    #################################################

    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None, args=args)
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None, args=args)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    weights[-1] = 1.0

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion  = MODEL.get_loss().cuda()
    criterion_pointcam = nn.MSELoss()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        # elif classname.find('Linear') != -1:
        #     torch.nn.init.xavier_normal_(m.weight.data)
        #     torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    ################
    sum_count  = 0
    sum_auroc  = 0.0
    sum_aupr   = 0.0
    sum_fpr    = 0.0
    sum_mIoU   = 0.0
    curve_auroc = []
    curve_aupr  = []
    curve_fpr   = []
    curve_mIoU  = []  
    ################

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('\n\n')
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen    = 0
        loss_sum      = 0
        loss_pointcam_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
                        
            seg_pred, trans_feat, _, _, _, _, attent1, attent2, attent3, w1, w2, w3 = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target      = target.view(-1, 1)[:, 0]

            #######################################################
            loss = criterion(seg_pred, target, trans_feat, weights)

            if args.use_pointcam:
                attent = w1*attent1 + w2*attent2 + w3*attent3

                target_attent = torch.zeros_like(target)
                idx = target == S3DISDataLoader.NEW_LABEL
                target_attent[idx] = 1.0
                target_attent = target_attent.to(torch.float32)

                loss_pointcam = criterion_pointcam(attent, target_attent)
                loss = loss + args.alpha*loss_pointcam
                loss_pointcam_sum += args.alpha*loss_pointcam
            #######################################################

            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen    += (BATCH_SIZE * NUM_POINT)
            loss_sum      += loss
            
        log_string('Training mean loss: %f' % (loss_sum / num_batches)) 
        log_string('Training mean loss_pointcam: %f' % (loss_pointcam_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen    = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            ##############
            seg_preds = []
            targets = []
            ##############

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
 
                seg_pred, trans_feat, seg_pred_msp, idx_msp, seg_pred_maxlogit, idx_maxlogit, attent1, attent2, attent3, w1, w2, w3 = classifier(points)

                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                ##########################################################              
                attent = w1*attent1 + w2*attent2 + w3*attent3

                if args.open_eval == 'pointcam':
                    seg_preds.append((1-attent).cpu().data.numpy())
                elif args.open_eval == 'msp':
                    seg_preds.append(seg_pred_msp.cpu().data.numpy())
                elif args.open_eval == 'maxlogit':
                    seg_preds.append(seg_pred_maxlogit.cpu().data.numpy()) 
                ##########################################################

                batch_label = target.cpu().data.numpy()
                target      = target.view(-1, 1)[:, 0]
                targets.append(target.cpu().data.numpy())

                pred_val = np.argmax(pred_val, 2)
                correct  = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen    += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            ########################
            # print open-set metrics
            if (epoch+1) % 1 == 0:
                auroc, aupr, fpr = eval_open_measure(np.array(seg_preds), np.array(targets), args, mask=None)
                log_string('eval point auroc: %f' % (auroc))
                log_string('eval point aupr: %f' % (aupr))
                log_string('eval point fpr: %f' % (fpr))

                curve_auroc.append(auroc*100)
                curve_avg_auroc = moving_average(np.array(curve_auroc), 5)

                curve_aupr.append(aupr*100)
                curve_avg_aupr = moving_average(np.array(curve_aupr), 5)

                curve_fpr.append(fpr*100)
                curve_avg_fpr = moving_average(np.array(curve_fpr), 5)

                mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
                if args.use_cutmix: 
                    mIoU = mIoU*(NUM_CLASSES)/(NUM_CLASSES-1)                    
                curve_mIoU.append(mIoU*100)
                curve_avg_mIoU = moving_average(np.array(curve_mIoU), 5)

                plt.title('AUROC Curve')
                plt.plot(np.arange(len(curve_avg_auroc)), curve_avg_auroc)
                plt.ylim(0, 100)
                plt.xlabel('Epoch')
                plt.ylabel('AUROC (%)')
                plt.savefig(os.path.join(str(checkpoints_dir), 'curve_avg_auroc.jpg'))
                plt.close()

                max_avg_auroc = np.amax(curve_avg_auroc)
                max_avg_aupr  = np.amax(curve_avg_aupr)
                min_avg_fpr   = np.amin(curve_avg_fpr)
                max_avg_mIoU  = np.amax(curve_avg_mIoU)              
                log_string('Best averaged auroc, aupr, fpr and mIoU are: %3f, %3f, %3f and %3f' % (max_avg_auroc, max_avg_aupr, min_avg_fpr, max_avg_mIoU))

                last_avg_auroc = curve_avg_auroc[-1]
                last_avg_aupr  = curve_avg_aupr[-1]
                last_avg_fpr   = curve_avg_fpr[-1]
                last_avg_mIoU  = curve_avg_mIoU[-1]              
                log_string('Last averaged auroc, aupr, fpr and mIoU are: %3f, %3f, %3f and %3f\n' % (last_avg_auroc, last_avg_aupr, last_avg_fpr, last_avg_mIoU))
            #####################################################################################################################################################

            # print IoU
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            if args.use_cutmix: 
                mIoU = mIoU*(NUM_CLASSES)/(NUM_CLASSES-1)     
            log_string('eval point avg class IoU: %f' % (mIoU))
            # log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            # log_string('eval point avg class acc: %f' % (
            #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
