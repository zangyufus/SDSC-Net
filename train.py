"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset

import hydra
import omegaconf
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics




seg_classes = {'AREA1': [0, 1, 2, 3, 4]}



seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}***************************
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def calculate_sem_IoU(pred_np, seg_np ,num_classes):
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    # original code: for sem_idx in range(seg_np.shape[0])
    for sem_idx in range(len(seg_np)):
        for sem in range(num_classes):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

@hydra.main(version_base="1.1",config_path='config', config_name='partseg')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)
    seed = int(args.get("seed", 2026))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info('Random seed: %d', seed)


    root = hydra.utils.to_absolute_path('D:/YYY-LSK/YYY_new/data/Vaihingen3D_AREA/')
   

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='val', normal_channel=args.normal)
    valDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)

    '''MODEL LOADING'''
 
    args.num_class = 5
    num_category = 1
    # # lsk-
    args.input_dim = (9 if args.normal else 6) + num_category
    
    num_part = args.num_class
    # shutil.copy(hydra.utils.to_absolute_path('models/{}/model光谱差异.py'.format(args.model.name)), '.')

    # getattr(object, name[, default])返回对象属性值
    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    

    if args.optimizer == 'Adam':
        decay_params = []
        no_decay_params = []

       

        for name, param in classifier.named_parameters():
            if not param.requires_grad:
                continue

            if no_decay_modules.intersection(name.split('.')):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.Adam(
            [
                {
                    'params': decay_params,
                    'weight_decay': args.weight_decay,
                },
                {
                    'params': no_decay_params,
                    'weight_decay': 0.0,
                },
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
        )

    # if args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #         classifier.parameters(),
    #         lr=args.learning_rate,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=args.weight_decay
    #     )
    # else:
    #     optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    log_dir = 'log_tensorboard_1'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer_train_loss = SummaryWriter(os.path.join(log_dir, 'train_loss')) #**********
    writer_train_accuracy = SummaryWriter(os.path.join(log_dir, 'train_acc'))
    writer_train_iou = SummaryWriter(os.path.join(log_dir, 'train_iou'))

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        train_loss = 0.0 #*******
        main_loss_sum = 0.0
        geometry_loss_sum = 0.0
        gate_sum = np.zeros(2, dtype=np.float64)
        count = 0.0 #********
        niter = 0.0 #********
        train_true_cls = [] #********
        train_pred_cls = [] #********
        train_true_seg = [] #********
        train_pred_seg = [] #********

        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        ###################################################
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, originxyz, label, target, cur_name) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            data = points.permute(0, 2, 1).float().cuda() #[16,9,2025] #***********
            batch_size = data.size()[0] #*******
            points = points.data.numpy()# 16 1024 6
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points) # 16 1024 6

            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda() # label:16 1; target:16 1024;
            # === 插入检查代码 ===
            if torch.isnan(points).any() or torch.isinf(points).any():
                print(f"警告：第 {i} 个 Batch 的输入点云数据包含 NaN 或 Inf！跳过该 Batch。")
                continue
            # ===================
            optimizer.zero_grad()

            seg_dgcnn_np = target.cpu().numpy()#*********

            model_input = torch.cat([
                points,
                to_categorical(label, num_category).repeat(1, points.shape[1], 1)
            ], -1)
            seg_pred, auxiliary = classifier(model_input, return_aux=True)
            pred_dgcnn = seg_pred.max(dim=2)[1] #***** [2,2048]

            seg_pred = seg_pred.contiguous().view(-1, num_part) # seg_pred:16384 50;
            target = target.view(-1, 1)[:, 0] # target:16384


            mask_invalid = (target < 0) | (target >= num_part)
            if mask_invalid.sum() > 0:
                target[mask_invalid] = 255
            # ==============================

            pred_choice = seg_pred.data.max(1)[1] # pred_choice:16384

            # # torch.eq(input, other, *, out=None):对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.num_point)) # append()方法向列表末尾追加元素
            main_loss = criterion(seg_pred, target)
            geometry_loss = criterion(
                auxiliary["geometry_logits"].reshape(-1, num_part), target
            )
            loss = main_loss + geometry_aux_weight * geometry_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            count += batch_size #*******
            train_loss += loss.item() * batch_size #*******
            main_loss_sum += main_loss.item() * batch_size
            geometry_loss_sum += geometry_loss.item() * batch_size
            gate_sum += auxiliary["gate_weights"].mean((0, 1)).cpu().numpy() * batch_size
            niter += batch_size #*******

            pred_dgcnn_np = pred_dgcnn.cpu().numpy()#************
            
            train_true_cls.append(seg_dgcnn_np.reshape(-1))  # (batch_size * num_points) #*******
            train_pred_cls.append(pred_dgcnn_np.reshape(-1))  # (batch_size * num_points) #*******
            train_true_seg.append(seg_dgcnn_np) #*******
            train_pred_seg.append(pred_dgcnn_np)  #*******

        train_true_cls = np.concatenate(train_true_cls) #*******
        train_pred_cls = np.concatenate(train_pred_cls) #*******
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls) #*******
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls) #*******
        train_true_seg = np.concatenate(train_true_seg, axis=0) #*******
        train_pred_seg = np.concatenate(train_pred_seg, axis=0) #*******
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg ,num_part) #*******
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                  train_loss * 1.0 / count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        logger.info(outstr) #*****
        logger.info('Main loss %.6f; geometry loss %.6f; gate G/S %s',
                    main_loss_sum / count, geometry_loss_sum / count,
                    (gate_sum / count).tolist())
        # Per-epoch parameter magnitudes make relation-path collapse observable.
        for module_name, module in classifier.named_modules():
            if (hasattr(module, "fc_gamma")
                    and getattr(module, "spectral_encoder", None) is not None):
                gamma_rms = module.fc_gamma[0].weight.detach().double().square().mean().sqrt().item()
                spectral_rms = module.spectral_encoder.mlp[0].weight.detach().double().square().mean().sqrt().item()
                logger.info('Attention %s: gamma_rms=%.3e spectral_rms=%.3e',
                            module_name, gamma_rms, spectral_rms)
        writer_train_loss.add_scalar('Train/loss', ((train_loss * 1.0) / count), epoch) #*******
        writer_train_accuracy.add_scalar('Train/accuracy', train_acc, epoch) #*****
        writer_train_iou.add_scalar('Train/mIOU', np.mean(train_ious), epoch) #*****

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {} 
            total_correct = 0
            total_seen = 0
            val_confusion = np.zeros((num_part, num_part), dtype=np.int64)
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, originxyz, label, target, cur_name) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                
                # data = points.permute(0, 2, 1).float() #***************
                # batch_size = data.size()[0] #**********

                seg_pred = classifier(torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1))
                
                # pred_dgcnn = seg_pred.max(dim=2)[1] #***********

                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)
                valid = (target >= 0) & (target < num_part)
                encoded = (target[valid].astype(np.int64) * num_part
                           + cur_pred_val[valid].astype(np.int64))
                val_confusion += np.bincount(
                    encoded, minlength=num_part * num_part
                ).reshape(num_part, num_part)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            b = float(np.sum((segl == l) | (segp == l)))
                            a = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            # test_metrics['class_avg_accuracy'] = np.mean(
            #     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            # 把 numpy 的 np.float 替换掉，同时解决 empty slice 或者除 0 问题
            seen_arr = np.array(total_seen_class, dtype=float)
            # 给零点贴块创可贴：只要总类别遇到数量0的，把除数变成极小数防止报 NaN
            seen_arr[seen_arr == 0] = 1e-6 
            test_metrics['class_avg_accuracy'] = np.mean(np.array(total_correct_class) / seen_arr)
            for cat in sorted(shape_ious.keys()):
                logger.info('eval block mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            intersection = np.diag(val_confusion).astype(np.float64)
            union = (val_confusion.sum(0) + val_confusion.sum(1)
                     - intersection)
            global_class_iou = np.divide(
                intersection, union, out=np.full(num_part, np.nan),
                where=union > 0,
            )
            global_miou = float(np.nanmean(global_class_iou))
            # Existing save conditions now use global validation mIoU.
            test_metrics['class_avg_iou'] = global_miou
            test_metrics['inctance_avg_iou'] = global_miou
            logger.info('Validation global class IoU: %s', global_class_iou)
            logger.info('Validation global mIoU: %.6f', global_miou)

        logger.info('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = 'best_model.pth'
            # absolute_path = os.path.abspath(savepath)  # 添加这行
            # logger.info('Saving at absolute path: %s' % absolute_path)  # 添加这行
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        if (test_metrics['accuracy'] >= best_acc):
            logger.info('Save accuracy model...')
            savepath = 'best_model_accuracy.pth'
            # absolute_path = os.path.abspath(savepath)  # 添加这行
            # logger.info('Saving at absolute path: %s' % absolute_path)  # 添加这行
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving accuracy model....')


        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        logger.info('Best accuracy is: %.5f' % best_acc)
        logger.info('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        logger.info('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        # Retain the latest model even when it does not beat the global best.
        torch.save({
            'epoch': epoch,
            'train_acc': train_instance_acc,
            'test_acc': test_metrics['accuracy'],
            'class_avg_iou': test_metrics['class_avg_iou'],
            'inctance_avg_iou': test_metrics['inctance_avg_iou'],
            'global_class_iou': global_class_iou.tolist(),
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'last_model.pth')
        global_epoch += 1

    writer_train_loss.close() #**********
    writer_train_accuracy.close() #*******
    writer_train_iou.close() #************

if __name__ == '__main__':
    #print(torch.cuda.is_available())
    main()