# Raw String level Fine-tuning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np 

import torch.distributed as dist
import torch.utils.data.distributed

from torch.optim import Adam
from transformers import get_cosine_schedule_with_warmup
import os
import argparse
import logging
import clip

from data.fashiongen_r import FashionGen
from utils.utils import RunningAverage, set_logger, seed_torch

# ===================================== HyperParameters =====================================
parser = argparse.ArgumentParser(description='Train FashionBert.')
# parser.add_argument('--data_root', help='path to FashionGen', default='./preprocess')
parser.add_argument('--save_path', help='path to checkppoint', default='OUTPUT-rCLIP')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--scratch', action='store_true')
parser.add_argument('--gpus', nargs='+', default=[0,1,2,3])
parser.add_argument('--load', action='store_true')
parser.add_argument('--dataset', type=str, default='FashionGen')
parser.add_argument('--select_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)

# Training parameters
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--num_warmup_steps', type=int, default=100)
parser.add_argument('--shuffle',action='store_true')
parser.add_argument('--train_select',action='store_true')
parser.add_argument('--eval',action='store_true')



args = parser.parse_args()
os.makedirs(args.save_path, exist_ok=True)


seed_torch(0)


def train_clip(model, dataloader, val_loader, new_added_keys,args):

    logging.info('Start Training ...')
    scaler = GradScaler(enabled=args.fp16)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Optimizer / Loss >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    if args.train_select:
        new_params = []
        for n, p in model.named_parameters():
            if n in new_added_keys:
                print(n + 'is added in new_params')
                new_params.append(p)

        # ------------ paras ------------
        new_params_id = list(map(id, new_params))
        old_params = filter(lambda p: id(p) not in new_params_id, model.parameters()) 
         # ------------ paras ------------

        optimizer = Adam( [   
        {'params': new_params},  
        {'params': old_params, 'lr': 0 }, 
        ],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4
        )

        # only train new params！
        for p in old_params:
            p.requires_grad = False
    
    else:
#          optimizer = Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=1e-4)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.epochs * len(dataloader))
    
    loss_fn = torch.nn.CrossEntropyLoss()

    model.float()
    best_acc = -999
    for epoch in range(args.epochs):
        model.train()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> log prepare >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        avg_total_loss = RunningAverage()
        img_acc = RunningAverage()
        txt_acc = RunningAverage()
        logging.info('Epoch {} ========================'.format(epoch))

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> one epoch >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for step, (image_input, texts, text_inputs_raw, (comp_str, category_str, brand_str, season_str, text)) in enumerate(dataloader):
            # image_input:      (bs, 3, 224, 224)
            # texts:       a list of string (without tag)
            # text_inputs_raw:  a list of string (concatenated with entities)
            # tags_id:         a list. element: long tensor,
        
            image_input = image_input.cuda()                                # (bs, 3, 224, 224)
            text_inputs = clip.tokenize(list(text_inputs_raw)).cuda()       # (bs, num_tokens)
            
            t1 =  clip.tokenize(list(category_str)).cuda()
            t2 = clip.tokenize(list(brand_str)).cuda()
            t3 = clip.tokenize(list(text)).cuda()
            t4 = clip.tokenize(list(season_str)).cuda()
            t5 = clip.tokenize(list(comp_str)).cuda()
            
            bs = image_input.shape[0]
            labels = torch.arange(bs).cuda()                # Contrastive Learning label (bs, ). 0,1,2,...bs-1

            with autocast(enabled=args.fp16):
                image_features, text_features, s1_f, s2_f, s3_f, s4_f, s5_f, t1_f, t2_f, t3_f, t4_f, t5_f = model(image_input, text_inputs, (t1,t2,t3,t4,t5), train = True)

                logits_per_image = image_features @ text_features.t()     # (bs, bs)
                logits_per_text = text_features @ image_features.t()      # (bs, bs)
                
                logits_per_image_1 = s1_f @ t1_f.t() 
                logits_per_text_1 = t1_f @ s1_f.t()
                
                logits_per_image_2 = s2_f @ t2_f.t() 
                logits_per_text_2 = t2_f @ s2_f.t()
                
                logits_per_image_3 = s3_f @ t3_f.t() 
                logits_per_text_3 = t3_f @ s3_f.t()
                
                logits_per_image_4 = s4_f @ t4_f.t() 
                logits_per_text_4 = t4_f @ s4_f.t()
                
                logits_per_image_5 = s5_f @ t5_f.t() 
                logits_per_text_5 = t5_f @ s5_f.t()
                
                loss_contra = 0.5 * (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels))
                loss_s1 =  0.5 * (loss_fn(logits_per_image_1, labels) + loss_fn(logits_per_text_1, labels))
                loss_s2 =  0.5 * (loss_fn(logits_per_image_2, labels) + loss_fn(logits_per_text_2, labels))
                loss_s3 =  0.5 * (loss_fn(logits_per_image_3, labels) + loss_fn(logits_per_text_3, labels))
                loss_s4 =  0.5 * (loss_fn(logits_per_image_4, labels) + loss_fn(logits_per_text_4, labels))
                loss_s5 =  0.5 * (loss_fn(logits_per_image_5, labels) + loss_fn(logits_per_text_5, labels))
                
                loss = loss_contra * 0.5 +  (loss_s1 + loss_s2 + loss_s3 + loss_s4 + loss_s5)/5 * 0.5
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()       # Updates the scale for next iteration  (FP16)
            lr_scheduler.step()

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Log >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            avg_total_loss.update(loss.cpu().item())
            img_acc.update((torch.max(logits_per_image.data, 1)[1] == labels).sum().item() / bs)
            txt_acc.update((torch.max(logits_per_text.data, 1)[1] == labels).sum().item() / bs)
            if step % 50 == 0:
                info1 = 'EPOCH: {}/{} | STEP: {}/{} | LR: {:05.8f} '.format(epoch, args.epochs, step, len(dataloader), lr_scheduler.get_lr()[0])
#                 info1 = 'EPOCH: {}/{} | STEP: {}/{} '.format(epoch, args.epochs, step, len(dataloader))
                info2 = 'Loss: {:05.4f}({:05.4f}) | '.format(loss.cpu().item(), avg_total_loss())
                info3 = 'Img CL Acc: {:05.4f} | Txt CL Acc: {:05.4f}'.format(img_acc(), txt_acc())
                logging.info(info1 + info2 + info3)

        logging.info('========================== Epoch End Summary ==========================')
        info1 = 'EPOCH: {}/{}  '.format(epoch, args.epochs)
        info2 = 'Loss: {:05.4f} | '.format(avg_total_loss())
        info3 = 'Img CL Acc: {:05.4f} | Txt CL Acc: {:05.4f}'.format(img_acc(), txt_acc())
        logging.info(info1 + info2 + info3)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> save model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model_name = os.path.join(args.save_path, "ckpt.pth.tar")

        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}, model_name)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluate >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        logging.info('Evaluation')
        metric = evaluate(model, val_loader)
        logging.info('Overall metric {}'.format(metric))
        if metric > best_acc:
            best_acc = metric
            logging.info('So far best model: {}. Metric: {}'.format(epoch, metric))
            torch.save({'state_dict': model.state_dict()}, os.path.join(args.save_path, "best.pth"))



def evaluate(model, val_loader):
    model.eval()
    img_feat_all = []
    text_feat_all = []
    with torch.no_grad():
        for step, (image_input, texts, text_inputs_raw, tags_id) in enumerate(val_loader):

            image_input = image_input.cuda()                            # (bs, 3, 224, 224)
            text_inputs = clip.tokenize(list(text_inputs_raw)).cuda()       # (bs, num_tokens)

            # ================================== given tags ==================================
#             image_features = model.module.encode_image(image_input)         # (B, 512)
#             text_features = model.module.encode_text(text_inputs)           # (B, 512)
            image_features = model.encode_image(image_input)         # (B, 512)
            text_features = model.encode_text(text_inputs)           # (B, 512)

            # normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            img_feat_all.append(image_features)
            text_feat_all.append(text_features)

    img_feat_all = torch.cat(img_feat_all, dim=0)       # (num_data, 512)
    text_feat_all = torch.cat(text_feat_all, dim=0)     # (num_data, 512)

    metric = evaluate_full_retrieval(img_feat_all, text_feat_all)
    return metric


def evaluate_(model, val_loader):
    model.eval()
    img_feat_all = []
    text_feat_all = []
    with torch.no_grad():
        for step, (image_input, texts, text_inputs_raw, tags_id) in enumerate(val_loader):

            image_input = image_input.cuda()                      # (bs, 3, 224, 224)
            text_inputs = clip.tokenize(list(texts)).cuda()       # (bs, num_tokens)

            # ================================== given tags ==================================
#             image_features = model.module.encode_image(image_input)         # (B, 512)
#             text_features = model.module.encode_text(text_inputs)           # (B, 512)
            image_features = model.encode_image(image_input)         # (B, 512)
            text_features = model.encode_text(text_inputs)           # (B, 512)

            # normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            img_feat_all.append(image_features)
            text_feat_all.append(text_features)

    img_feat_all = torch.cat(img_feat_all, dim=0)       # (num_data, 512)
    text_feat_all = torch.cat(text_feat_all, dim=0)     # (num_data, 512)

    metric = evaluate_full_retrieval(img_feat_all, text_feat_all)
    return metric

def evaluate_full_retrieval(img_feat_all, text_feat_all):
    total = img_feat_all.shape[0]
    # --------------------- Image to Text ---------------------
    rk1, rk5, rk10 = 0, 0, 0
    for i in range(img_feat_all.shape[0]):
        similarity_im2txt = img_feat_all[i:i+1] @ text_feat_all.T   # (1, num_data)
        _, ind = similarity_im2txt[0].sort(descending=True)
        ind = ind.cpu().tolist()
        rk1 = rk1 + 1 if i in ind[:1] else rk1
        rk5 = rk5 + 1 if i in ind[:5] else rk5
        rk10 = rk10 + 1 if i in ind[:10] else rk10
    logging.info('Image to Text: R@1: {:05.3f} R@5: {:05.3f} R@10: {:05.3f}'.format(rk1/total, rk5/total, rk10/total))
    metric = rk1/total + rk5/total + rk10/total

    # --------------------- Text to Image ---------------------
    rk1, rk5, rk10 = 0, 0, 0
    for i in range(text_feat_all.shape[0]):
        similarity_im2txt = text_feat_all[i:i+1] @ img_feat_all.T   # (num_data, 1)
        _, ind = similarity_im2txt[0].sort(descending=True)
        ind = ind.cpu().tolist()
        rk1 = rk1 + 1 if i in ind[:1] else rk1
        rk5 = rk5 + 1 if i in ind[:5] else rk5
        rk10 = rk10 + 1 if i in ind[:10] else rk10
    logging.info('Text to Image: R@1: {:05.3f} R@5: {:05.3f} R@10: {:05.3f}'.format(rk1/total, rk5/total, rk10/total))
    metric = rk1 / total + rk5 / total + rk10 / total + metric
    return metric

def test(model, val_loader):
    metric = evaluate(model, val_loader)
    metric = evaluate_(model, val_loader)
    return metric

if __name__ == '__main__':
    os.makedirs(args.save_path, exist_ok=True)
    set_logger(os.path.join(args.save_path, 'training.log'))
    
    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logging.info(message)
    
    # ===================================== loda CLIP =====================================
    logging.info('Create Model .....')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info('Device: ' + device)
    assert device == "cuda:0"
    
    model_clip, preprocess = clip.load('ViT-B/32', device=device, jit=False)

    # load CLIP parameters
#     state_dict = torch.load('./OUTPUT-CLIP/best.pth')
#     model_clip.load_state_dict(state_dict['state_dict'])
#     model_clip = model_clip.cuda()
#     model_clip.load_state_dict(torch.load('../CLIP/models/clip_ft/clip_ft_E10'))
    
    # ===================================== Load rCLIP =====================================
    #from rclip.rclip_v2_naive import rCLIP
    from rclip.rclip_v2_5 import rCLIP
    model_rclip = rCLIP(device=device)
    
    # load weights of CLIP
    rclip_dict = model_rclip.state_dict()
    clip_dict = model_clip.state_dict() 
    model_rclip.load_state_dict(clip_dict, strict=False)     

    
    # load pre-trained selection embeddings if FALSE
    if args.train_select == False:
        if args.eval:
            pass
        else:
            state_dict = torch.load(args.select_path+'/best.pth')
            model_rclip.load_state_dict(state_dict['state_dict'])
            pass # issue#1: ablate it for co-train (naive)
    else:
        pass
    
    # ===================================== load previous model =====================================
    if args.load:
        state_dict = torch.load('./OUTPUT-rCLIP_v2_shu-selected/best_5_5e-4.pth')
        model_rclip.load_state_dict(state_dict['state_dict'])
    
    # ===================================== paras filter =====================================
    rclip_keys = model_rclip.state_dict().keys()
    clip_keys = model_clip.state_dict().keys()
    
    new_added_keys = rclip_keys - clip_keys
    
#     # load trained model
#     state_dict = torch.load('./OUTPUT-rCLIP-play/best-20.pth')
#     model_rclip.load_state_dict(state_dict['state_dict'])
    
    # to CUDA
    model_rclip.float()
    model_rclip = model_rclip.cuda()

    # ===================================== Data Loader =====================================
    logging.info('Create Dataset ......')
    
    if args.dataset == 'FashionGen':
        data_root = './preprocess'
        train_data = FashionGen(data_root=data_root, mode='train')
        val_data = FashionGen(data_root=data_root, mode='validation')
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, num_workers=16, pin_memory=False, shuffle=args.shuffle)
    
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=False, num_workers=16, pin_memory=False)

    # ===================================== single model =====================================
   
    if args.eval:
        # load trained model
        state_dict = torch.load(args.model_path)
        model_rclip.load_state_dict(state_dict['state_dict'])
        test(model_rclip, val_loader)
    
    else:
        train_clip(model_rclip, train_loader, val_loader, new_added_keys, args)

