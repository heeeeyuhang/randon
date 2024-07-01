import os
from threading import local

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr

#在一个训练周期内对模型进行训练和验证，并记录损失和准确率，同时根据配置保存模型权重
"""
model_train 和 model：训练模型和模型本身，通常用于训练和评估阶段。
loss_history：记录损失值的对象。
optimizer：优化器，用于更新模型参数。
epoch：当前的 epoch 号。
epoch_step 和 epoch_step_val：训练和验证步骤数。
gen 和 gen_val：训练和验证的数据生成器。
Epoch：总的 epoch 数。
cuda：是否使用 CUDA 进行加速。
fp16：是否使用半精度浮点数。
scaler：用于混合精度训练的比例器。
save_period 和 save_dir：保存模型的周期和目录。
local_rank：用于分布式训练。"""
def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):#
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0
    val_accuracy    = 0

    if local_rank == 0:
        print('Start Train')#在 local_rank 为 0 时打印训练开始信息并初始化进度条
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()#开始训练
    for iteration, batch in enumerate(gen):#循环在每个训练数据
        if iteration >= epoch_step: #迭代数超过 epoch_step 则停止
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)#放入gpu
                targets = targets.cuda(local_rank)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:#根据是否使用半精度浮点数执行不同的前向传播和反向传播
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss_value.item()
        #开始使用测试集评估
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()

            outputs     = model_train(images)
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            
            val_loss    += loss_value.item()
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy    += accuracy.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
