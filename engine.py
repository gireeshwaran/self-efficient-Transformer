# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import accuracy, ModelEma
import utils
import torchvision

def save_rec_out_batch(org_im_batch, rec_im_batch, masks, x_cls_recons, epoch, args, trainin_stat):
    if args.output_dir:
        print_out = args.output_dir + '/rec_img'  + '/' + trainin_stat + '/img_' + str(epoch).zfill(4) + '.jpg'
        
        
        image_list = org_im_batch[0:min(16, len(org_im_batch))].cpu()
        full_recons = torch.zeros_like(org_im_batch)
        for i in range(4):
            masks1 = masks[i].reshape(masks.size()[1], 14, 14).repeat_interleave(16, 1).repeat_interleave(16, 2).unsqueeze(1).contiguous()
            masks1 = masks1.repeat(1, 3, 1, 1)
            full_recons[masks1==1] = rec_im_batch[i][masks1==1]
            
            image_list = torch.cat([image_list, 
                                    (org_im_batch.detach().clone()*(1-masks1))[0:min(16, len(org_im_batch))].cpu(), 
                                    rec_im_batch[i][0:min(16, len(org_im_batch))].cpu()], dim=0)
        
        image_list = torch.cat([image_list, full_recons[0:min(16, len(org_im_batch))].cpu()], dim=0)
        image_list = torch.cat([image_list, x_cls_recons[0:min(16, len(org_im_batch))].cpu()], dim=0)
        torchvision.utils.save_image(image_list, print_out, nrow=min(16, len(org_im_batch)), normalize=True, range=(-1, 1))
        
        
        
        #torchvision.transforms.ToPILImage()(mask[0].reshape(14, 14).sub(-1).div(max(2, 1e-5))).convert("RGB").save('mask'+str(i).zfill(3)+'.png')


def train_one_epoch_rec(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, args = None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    plot_ = True

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs_cls, loss_rec, recons_images, mask, x_cls_recons  = model(samples, recHead = True)
            loss_cls = criterion(outputs_cls, targets)
            
            if epoch  < 250:
                m_cls = 0
                m_rec = 1
            elif epoch >= 250 and epoch < 400:
                m_cls = 1
                m_rec = 1
            else:
                m_cls = 1
                m_rec = 0
                
            loss  = m_cls*loss_cls +m_rec*loss_rec

        loss_value = loss.item()

        if plot_ and (epoch % 10 == 0):
                plot_ = False
                save_rec_out_batch(samples, recons_images, mask, x_cls_recons, epoch, args, "train")
    
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(rec_mult=args.balancingRecLoss[epoch])
        metric_logger.update(loss_rec=loss_rec.item())
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_rec(data_loader, model, device, epoch, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    plot_ = True
    for images, target in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output_cls, loss_rec, recons_images, mask, x_cls_recons = model(images, recHead = True)
            
            loss_cls = criterion(output_cls, target)

            loss = loss_cls + loss_rec
            #save rec only onces 
            if plot_:
                plot_ = False
                save_rec_out_batch(images, recons_images, mask, x_cls_recons, epoch, args, "test")

        acc1, acc5 = accuracy(output_cls, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_rec=loss_rec.item())
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}