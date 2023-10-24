# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops.opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
import torch.nn.functional as F

best_prec1 = 0
best_test_prec1 = 0
best_epoch = -1


class RelationClassifier(torch.nn.Module):
    def __init__(self, num_class, dim_in, dim_out):
        super(RelationClassifier, self).__init__()
        self.head = torch.nn.Sequential(
                          torch.nn.Linear(dim_in, dim_out),
                          torch.nn.ReLU(inplace=True),
                          torch.nn.Linear(dim_out, num_class),
                          )

    def forward(self, x):
        bz, nr, dim = x.shape
        x = x.view(bz*nr, dim)
        z = self.head(x)
        z = z.view(bz, nr, z.shape[-1])
        return z

class Classifier(torch.nn.Module):
    def __init__(self, num_class, dim_in):
        super(Classifier, self).__init__()
        self.head = torch.nn.Linear(dim_in, num_class)

    def forward(self, x):
        bz, nr, dim = x.shape
        x = x.view(bz*nr, dim)
        z = self.head(x)
        z = z.view(bz, nr, z.shape[-1])
        return z

def entropy(predictions, dim=2, reduction='none'):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=dim)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

def my_loss(criterion, output, target, attn_list, relation_output):
    # ce
    ce = criterion(output, target)
    # entropy minimization
    ent = 0
    for attn in attn_list:
        ent += entropy(attn, dim=2).mean()
    ent /= len(attn_list)
    # mean entropy maximization
    kl = 0
    for attn in attn_list:
        attn = attn.mean(dim=1)
        kl += -entropy(attn, dim=1).mean()
    kl /= len(attn_list)
    # cross-relation ce
    num_relation = relation_output.shape[1]
    ## construct target
    relation_target = [torch.ones_like(target)*r for r in range(num_relation)]
    relation_target = torch.stack(relation_target, dim=1)
    relation_target = target.unsqueeze(1) * num_relation + relation_target
    ## calculate loss
    relation_ce = criterion(relation_output.view(-1, relation_output.shape[-1]), relation_target.flatten())

    loss = ce + ent * args.entropy_min + kl * args.entropy_mean_max + relation_ce * args.cross_relation
    loss_dict = {'ce': ce.item(), 'min_entropy': ent.item(), 'max_mean_entropy': kl.item(), 
                 'cross_relation': relation_ce.item()}
    return loss, loss_dict


def main():
    global args, best_prec1, best_test_prec1, best_epoch
    args = parser.parse_args()


    # set seed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False


    num_class, args.train_list, args.val_list, args.test_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['main', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print(args)
    print('storing name: ' + args.store_name)

    check_folders()

    args.mix_layer_names = ['layer'+name for name in args.mix_layers]
    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                local_feature_dim=args.local_feature_dim,
                global_feature_dim=args.global_feature_dim,
                num_spatial_group=args.num_spatial_group, 
                num_channel_group=args.num_channel_group, 
                aggregator_dim=args.aggregator_dim, 
                sgm_temp=args.sgm_temp, 
                mix_layers=args.mix_layer_names, mix_alpha=args.mix_alpha, mix_prob=args.mix_prob,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=args.fc_lr5)
    relation_classifier = RelationClassifier(num_class=num_class*(args.num_segments-1), 
                                             dim_in=args.img_feature_dim, 
                                             dim_out=args.relation_classifier_feat_dim)

        
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    # append the parameters of relation classifier 
    policies.append({'params': relation_classifier.parameters(), 'lr_mult': args.relation_classifier_lr, 
                     'decay_mult': args.relation_classifier_lr, 'name': "relation_classifier"},)
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    relation_classifier = torch.nn.DataParallel(relation_classifier, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, 
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.test_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_training = open(os.path.join('save', args.dataset, args.store_name, 'seed{}'.format(args.seed), '{}.log'.format(timestamp)), 'w')
    log_training.write(str(args) + '\n')
    log_training.flush()
    # with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
    #     f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join('save', args.dataset, args.store_name, 'seed{}'.format(args.seed), 'tensorboard'))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, relation_classifier, criterion, optimizer, epoch, log_training, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)
            test_prec1 = test(test_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = prec1
                best_test_prec1 = test_prec1
                best_epoch = epoch 
            tf_writer.add_scalar('acc/val_top1_best', best_prec1, epoch)
            tf_writer.add_scalar('acc/test_top1_best', best_test_prec1, epoch)

            output_best = 'Best Prec@1: %.6f and Corresponding Test Prec@1: %.6f (in epoch %d)\n' % (best_prec1, best_test_prec1, best_epoch)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, relation_classifier, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_ent = AverageMeter()
    losses_kl = AverageMeter()
    losses_cross = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output, feature, attn = model(input)
        output = output.sum(dim=1)
        loss, loss_dict = my_loss(criterion, output, target, attn, relation_classifier(feature))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_ce.update(loss_dict['ce'], input.size(0))
        losses_ent.update(loss_dict['min_entropy'], input.size(0))
        losses_kl.update(loss_dict['max_mean_entropy'], input.size(0))
        losses_cross.update(loss_dict['cross_relation'], input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'CE {loss_ce.val:.4f} ({loss_ce.avg:.4f})\t'
                      'MinEntropy {loss_ent.val:.4f} ({loss_ent.avg:.4f})\t'
                      'MaxMeanEntropy {loss_kl.val:.4f} ({loss_kl.avg:.4f})\t'
                      'CrossRelationCE {loss_cross.val:.4f} ({loss_cross.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_ce=losses_ce, loss_ent=losses_ent, loss_kl=losses_kl, loss_cross=losses_cross, 
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output, _, _ = model(input)
            output = output.sum(dim=1)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Validation: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Validation Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/val', losses.avg, epoch)
        tf_writer.add_scalar('acc/val_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/val_top5', top5.avg, epoch)

    return top1.avg


def test(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output, _, _ = model(input)
            output = output.sum(dim=1)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Test Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best):
    filename = 'save/%s/%s/%s/ckpt_latest.pth.tar' % (args.dataset, args.store_name, 'seed{}'.format(args.seed))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('latest.pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_folders():
    """Create log and model folder"""
    folders_util = [
        os.path.join('save', args.dataset), 
        os.path.join('save', args.dataset, args.store_name),
        os.path.join('save', args.dataset, args.store_name, 'seed{}'.format(args.seed)),
        os.path.join('save', args.dataset, args.store_name, 'seed{}'.format(args.seed), 'tensorboard'),
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
