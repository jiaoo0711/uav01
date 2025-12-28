# coding=utf-8
from __future__ import (absolute_import, division, unicode_literals)

import os
import sys

import time
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim, distributed
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from params import get_args, save_hp_to_json
from modules import CLIP4Clip, convert_weights
from modules import SimpleTokenizer as ClipTokenizer
from modules.file import PYTORCH_PRETRAINED_BERT_CACHE
from dataloaders.data_dataloaders import DATALOADER_DICT
from utils.lr_scheduler import lr_scheduler
from utils.optimization import BertAdam, prep_optim_params_groups
from utils.log import setup_primary_logging, setup_worker_logging
from utils.misc import set_random_seed, convert_models_to_fp32, save_checkpoint
from utils.dist_utils import is_master, get_rank, is_dist_avail_and_initialized, init_distributed_mode
from utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

best_R1 = 0

def main(args):
    """main function"""

    # Sync max_words to fg_clip_max_len if using FG-CLIP for robustness
    if args.use_fgclip and args.max_words != args.fg_clip_max_len:
        logging.warning(f"Parameter mismatch in FG-CLIP mode. Syncing 'max_words' ({args.max_words}) to 'fg_clip_max_len' ({args.fg_clip_max_len}).")
        args.max_words = args.fg_clip_max_len

    set_random_seed(args.seed)

    # Set multiprocessing type to spawn.
    if not args.remote:
        torch.multiprocessing.set_start_method('spawn')

    # Set logger
    log_queue = setup_primary_logging(os.path.join(args.output_dir, "log.txt"), args.log_level, args.remote)

    # lmdb
    if args.lmdb_dataset not in [None, 'None']:
        assert os.path.exists(args.lmdb_dataset)
        print('INFO: [dataset] Using {} as data source'.format(args.lmdb_dataset))

    # the number of gpus
    args.ngpus_per_node = torch.cuda.device_count()
    print("INFO: [CUDA] The number of GPUs in this node is {}".format(args.ngpus_per_node))

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        if args.remote:
            raise NotImplementedError
        else:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, log_queue, args))
    else:
        # nn.DataParallel (DP)
        if args.dp:
            args.gpu, args.world_size = args.multigpu[0], len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


def main_worker(gpu, ngpus_per_node, log_queue, args):
    """main worker"""
    global best_R1
    args.gpu = gpu

    ## ####################################
    # initilization
    ## ####################################
    global_rank = init_distributed_mode(args, ngpus_per_node, gpu)
    setup_worker_logging(global_rank, log_queue, args.log_level)
    # Lock the random seed of the model to ensure that the model initialization of each process is the same.
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # save parameters
    if is_master(): save_hp_to_json(args.output_dir, args)


    ## ####################################
    # create model
    ## ####################################
    # create tokenizer
    if args.use_fgclip:
        # 使用FG-CLIP的tokenizer
        from transformers import AutoTokenizer, AutoConfig
        import os
        
        tokenizer_path_to_load = args.fgclip_model_name
        tokenizer_kwargs = {}

        if args.fgclip_local_path:
            possible_paths = [
                args.fgclip_local_path,
                os.path.abspath(args.fgclip_local_path),
                # 根据您的项目结构，可能需要调整这些相对路径
                os.path.join(os.path.dirname(__file__), "..", "FG-CLIP", "pretrained", "fg-clip-base"), 
                os.path.join(os.getcwd(), "FG-CLIP", "pretrained", "fg-clip-base"), # 假设FG-CLIP与STOP在同一级目录
                "../FG-CLIP/pretrained/fg-clip-base", 
                "../../FG-CLIP/pretrained/fg-clip-base"
            ]
            
            found_path = None
            for path_option in possible_paths:
                # 检查config.json和tokenizer.json是否存在，这是加载本地模型的更可靠方式
                if os.path.exists(path_option) and os.path.exists(os.path.join(path_option, "config.json")) and os.path.exists(os.path.join(path_option, "tokenizer.json")):
                    found_path = path_option
                    print(f"INFO: [Tokenizer] Found tokenizer and config at: {found_path}")
                    break
            
            if found_path:
                print(f"INFO: [Tokenizer] Loading tokenizer from local path: {found_path} with local_files_only=True")
                tokenizer_path_to_load = found_path
                tokenizer_kwargs['local_files_only'] = True
            else:
                print(f"WARN: [Tokenizer] Local path {args.fgclip_local_path} provided but not found or incomplete. Falling back to Hugging Face model name: {args.fgclip_model_name}")
        
        print(f"INFO: [Tokenizer] Attempting to load tokenizer: {tokenizer_path_to_load}")
        try:
            # 尝试在加载时就设置 model_max_length，但更推荐的方式是加载后再显式设置，因为 from_pretrained 可能不会直接覆盖 config 中的值
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_to_load, **tokenizer_kwargs)
            
            # 显式设置 tokenizer 的 model_max_length
            # 确保 args.fg_clip_max_len 被正确使用
            if hasattr(args, 'fg_clip_max_len') and args.fg_clip_max_len > 0:
                tokenizer.model_max_length = args.fg_clip_max_len
                print(f"INFO: [Tokenizer] Explicitly set tokenizer.model_max_length to: {tokenizer.model_max_length} (from args.fg_clip_max_len)")
            else:
                # 如果命令行没有指定，尝试从config中获取，如果config中没有，则使用一个备用值或FG-CLIP的常见值
                try:
                    config = AutoConfig.from_pretrained(tokenizer_path_to_load, **tokenizer_kwargs)
                    if hasattr(config, 'text_config') and hasattr(config.text_config, 'max_position_embeddings'): # FG-CLIP可能嵌套配置
                       tokenizer.model_max_length = config.text_config.max_position_embeddings
                    elif hasattr(config, 'max_position_embeddings'): # 标准CLIP或类似模型的配置
                       tokenizer.model_max_length = config.max_position_embeddings
                    else: # FG-CLIP默认值
                       tokenizer.model_max_length = 248 
                    print(f"INFO: [Tokenizer] Set tokenizer.model_max_length from config or default: {tokenizer.model_max_length}")
                except Exception as e:
                    tokenizer.model_max_length = 248 # FG-CLIP 默认值
                    print(f"WARN: [Tokenizer] Could not determine model_max_length from config ({e}), defaulting to {tokenizer.model_max_length}")

        except Exception as e:
            print(f"ERROR: [Tokenizer] Failed to load tokenizer '{tokenizer_path_to_load}'. Error: {e}")
            print(f"ERROR: [Tokenizer] Please ensure that the path is correct or the model name is valid on Hugging Face Hub.")
            print(f"ERROR: [Tokenizer] If using a local path, ensure it contains tokenizer.json, config.json and other necessary files.")
            raise

        print(f"INFO: [Tokenizer] Tokenizer loaded. Type: {type(tokenizer)}")
        print(f"INFO: [Tokenizer] tokenizer.model_max_length: {tokenizer.model_max_length}")
        print(f"INFO: [Tokenizer] tokenizer.padding_side: {tokenizer.padding_side}")
        print(f"INFO: [Tokenizer] tokenizer.name_or_path: {tokenizer.name_or_path}")

    else:
        # 使用原始CLIP tokenizer
        tokenizer = ClipTokenizer()
    model_state_dict = torch.load(args.init_model, map_location='cpu') if args.init_model else None
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    
    model = CLIP4Clip.from_pretrained(args.cross_model, 
                                        cache_dir=cache_dir,
                                        state_dict=model_state_dict,
                                        task_config=args)
   
    # Set the model to train mode before passing parameters to the optimizer
    model.train()
   
    #model.freeze_cip_layers(args.freeze_layer_num)
    
    #logging.info('\nweight from DeepCluster')
     

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp"or args.gpu is None:  	# or args.precision == "fp32" 
        logging.info("[weight convert] ==>> Convert weights to fp32 for {}...".format(args.precision))
        convert_models_to_fp32(model)
        logging.info("[weight convert] ==>> Convert done!")

    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                find_unused_parameters=False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", get_rank() % ngpus_per_node)


    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT
    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs


    ## ####################################
    # optimization strategies
    ## ####################################
    optimizer_grouped_parameters = prep_optim_params_groups(args, model, coef_lr=args.coef_lr)
    scaler = GradScaler() if args.precision == "amp" else None
    if args.optim == 'BertAdam':
        logging.info('[optimizer] Using BertAdam Optimizer...')
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                                schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                                t_total=num_train_optimization_steps, weight_decay=args.wd,
                                max_grad_norm=1.0)
        scheduler = None
    elif args.optim == 'AdamW':
        logging.info('[optimizer] Using AdamW Optimizer...')
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.wd)
        scheduler = lr_scheduler(mode='cos', init_lr=args.lr, all_iters=num_train_optimization_steps,
                                    slow_start_iters=args.warmup_proportion * num_train_optimization_steps,
                                    weight_decay=args.wd
                                )
    else:
        raise NotImplementedError

    if is_master():
        tf_writer = SummaryWriter(args.tensorboard_path)
    else:
        tf_writer = None

    ## ####################################
    #  optionally resume from a checkpoint
    ## ####################################
    start_epoch, global_step = 0, 0
    if args.resume and os.path.exists(args.resume):
        logging.info("Loading checkpoint from {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict']) # consume_prefix_in_state_dict_if_present
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        
        start_epoch = checkpoint['epoch'] + 1
        best_R1 = checkpoint['best_R1']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.precision == "amp":
            assert scaler is not None, "Found 'scaler' is None, please check!"
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logging.info("Loaded checkpoint from successfully.")
        del checkpoint

    # load from pretrained
    if args.load_from_pretrained and args.pretrained_dir:
        logging.info("Loading checkpoint from {}".format(args.pretrained_dir))
        checkpoint = torch.load(args.pretrained_dir, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        logging.info("Loaded checkpoint from successfully.")
        del checkpoint


    ## ####################################
    # train and evalution
    ## ####################################
    if is_master():
        logging.info("\n======================== Running training ========================")
        logging.info("  Num examples = %d", train_length)
        logging.info("  Batch size = %d", args.batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
        logging.info("\n======================== Running test ========================")
        logging.info("  Num examples = %d", test_length)
        logging.info("  Batch size = %d", args.batch_size_val)
        logging.info("  Num steps = %d", len(test_dataloader))
        logging.info("\n======================== Running val ========================")
        logging.info("  Num examples = %d", val_length)

    all_start = time.time()

    if args.do_eval and is_master():
        R1, infer_epoch_time, info_str = eval_epoch(model, test_dataloader, device, args=args)
        torch.cuda.synchronize()
        all_time = time.time() - all_start
        logging.info('The total running time of the program is {:.2f} Seconds\n'.format(all_time))
        logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
                        torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
        sys.exit(0)

    eval_infer_times = []
    best_e = 0
    best_info = []
    for epoch in range(start_epoch, args.epochs):
        if is_dist_avail_and_initialized():
            train_sampler.set_epoch(epoch)
        # set_random_seed(epoch + args.seed)

        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, optimizer,
                                            global_step, scaler=scaler, tf_writer=tf_writer, scheduler=scheduler)

        if is_master():
            logging.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

            # Run on val dataset, this process is *TIME-consuming*.
            R1, infer_epoch_time, info_str = eval_epoch(model, test_dataloader, device, args=args, epoch=epoch)
            eval_infer_times.append(infer_epoch_time)
            if best_R1 <= R1:
                best_R1 = R1
                best_e = epoch
                best_info = info_str
            logging.info("The best R1 is: {:.4f}, best_e={}\n".format(best_R1, best_e))
            # save checkpoint
            ckpt_dict = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': 'CLIp4Clip',
                    'state_dict': model.state_dict(),
                    'best_acc1': best_R1,
                    'optimizer': optimizer.state_dict(),
                }
            if scaler is not None: ckpt_dict['scaler'] = scaler.state_dict()
            save_checkpoint(ckpt_dict, best_R1 <= R1, args.output_dir, filename='ckpt.pth.tar')

    all_time = time.time() - all_start

    if is_master():
        logging.info('The total running time of the program is {:.1f} Hour {:.1f} Minute\n'.format(all_time // 3600, 
                    all_time % 3600 / 60))
        logging.info('The average inference time of {} runs is {:.2f} Seconds\n'.format(args.epochs, np.mean(eval_infer_times)))
        logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
                    torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
        logging.info("The best R1 is: {:.4f}, best_epoch={}\n".format(best_R1, best_e))
        for info in best_info:
            logging.info(info)
        print("The above program id is {}\n".format(args.output_dir))

    torch.cuda.empty_cache()
    sys.exit(0)


def train_epoch(epoch, args, model, train_dataloader, device, optimizer, global_step,
                scheduler=None, scaler=None, tf_writer=None):
    samples_per_epoch = len(train_dataloader.dataset)

    # torch.cuda.empty_cache()
    model.train()

    if epoch == 0 and is_master():
        no_clip = args.new_added_modules
        trainable_size =0
        total_param_size  = 0  
        for name, param in model.module.named_parameters():
            if param.requires_grad==True:
                total_param_size += param.numel() 
                trainable_size += param.numel() 
                param_size_MB = param.numel()/(1000**2)
                # logging.info(f'trainerble parameters are: {name}, size is {param_size_MB:.4f} MB')
            else:
                total_param_size += param.numel() 
        trainable_size_MB = trainable_size/(1000**2)
        total_param_size_MB = total_param_size/(1000**2)
        percentage = (trainable_size / total_param_size)*100
        # logging.info("Trainable param percentage are: {}".format(percentage))
        # logging.info("Trainable params are: {} MB, Total params are: {} MB".format(trainable_size_MB,total_param_size_MB))


    total_loss = 0

    end = time.time()

    for step, batch in enumerate(train_dataloader):
        # if step == 1:
        #     break
        # logging.info(f"Step: {step}   Batch: {batch}")
        
        optimizer.zero_grad()
        if scheduler is not None: scheduler(optimizer, global_step=global_step)
        
        # Move batch to device (CUDA or CPU only, no NPU support)
        if torch.cuda.is_available():
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        
        # Check if using FG-CLIP with long/short text inputs
        if args.use_fgclip and len(batch) == 8:
            # FG-CLIP mode: batch contains (input_ids_long, attention_mask_long, segment_ids_long,
            #                              input_ids_short, attention_mask_short, segment_ids_short,
            #                              video, video_mask)
            input_ids_long, attention_mask_long, segment_ids_long, \
            input_ids_short, attention_mask_short, segment_ids_short, \
            video, video_mask = batch
            
            # Store batch size for logging
            batch_size = len(input_ids_long)
            
            data_time = time.time() - end
            
            # forward with FG-CLIP multi-loss
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type, enabled=scaler is not None):
                output = model(
                    input_ids=None, token_type_ids=None, attention_mask=None,
                    video=video, video_mask=video_mask,
                    input_ids_long=input_ids_long,
                    attention_mask_long=attention_mask_long,
                    segment_ids_long=segment_ids_long,
                    input_ids_short=input_ids_short,
                    attention_mask_short=attention_mask_short,
                    segment_ids_short=segment_ids_short
                )
                loss = output['loss'].mean()
                sim_loss = output.get('sim_loss', loss).mean()
                loss_itcl = output.get('loss_itcl', None)
                loss_itcs = output.get('loss_itcs', None)
        else:
            # Original mode: single text input
            input_ids, input_mask, segment_ids, video, video_mask = batch
            
            # Store batch size for logging
            batch_size = len(input_ids)
            
            data_time = time.time() - end

            # forward
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type, enabled=scaler is not None):
                output = model(input_ids, segment_ids, input_mask, video, video_mask)
                loss = output['loss'].mean()
                sim_loss = output['sim_loss'].mean()
                loss_itcl = None
                loss_itcs = None

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            
        # logging.info(f"Step: {step}   Loss: {loss}")
        # logging.info(f"Step: {step}   Sim_Loss: {sim_loss}")

        # update weights
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        try:
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, 0, 4.6052)
            else:
                torch.clamp_(model.clip.logit_scale.data, 0, 4.6052)
        except AttributeError:
            logging.warning("logit_scale not found for clamping.")


        batch_time = time.time() - end
        end = time.time()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            global_step += 1
            if global_step % args.n_display == 0 and is_master():
                # batch_size is already defined in both code paths above
                num_samples = (step + 1) * batch_size * args.world_size
                percent_complete = num_samples * 1.0 / samples_per_epoch * 100
                
                try:
                    if hasattr(model, 'module'):
                        logit_scale_val = model.module.clip.logit_scale.data.item()
                    else:
                        logit_scale_val = model.clip.logit_scale.data.item()
                except AttributeError:
                    logit_scale_val = float('nan')

                lr_tmp = optimizer.param_groups[0]['lr'] if args.optim == 'AdamW' else \
                            optimizer.get_lr()[0]
                
                if args.use_fgclip and loss_itcl is not None and loss_itcs is not None:
                    logging.info(
                        f"Epoch: {epoch} [{num_samples} ({percent_complete:.1f}%)]\\t"
                        f"Loss: {loss.item():.4f} (ITCL: {loss_itcl.item():.4f}, ITCS: {loss_itcs.item():.4f}) \\t"
                        f"Data (t) {data_time:.3f}\\tBatch (t) {batch_time:.3f}"
                        f"\\tLR: {lr_tmp:.1e}\\tlogit_scale {logit_scale_val:.3f}"
                    )
                else:
                    logging.info(
                        f"Epoch: {epoch} [{num_samples} ({percent_complete:.1f}%)]\\t"
                        f"SimLoss: {sim_loss.item():.4f} \\t"
                        f"Data (t) {data_time:.3f}\\tBatch (t) {batch_time:.3f}"
                        f"\\tLR: {lr_tmp:.1e}\\tlogit_scale {logit_scale_val:.3f}"
                    )
                # tensorboard log
                log_data = {
                    "sim_loss": sim_loss.item(),
                    "data_time": data_time,
                    "batch_time": batch_time,
                    "scale": logit_scale_val,
                    "lr": lr_tmp
                }
                if args.use_fgclip and loss_itcl is not None and loss_itcs is not None:
                    log_data["loss_itcl"] = loss_itcl.item()
                    log_data["loss_itcs"] = loss_itcs.item()
                for name, val in log_data.items():
                    name = "train/" + name
                    if tf_writer is not None:
                        tf_writer.add_scalar(name, val, global_step=global_step)

        total_loss += float(loss)

    total_loss = total_loss / len(train_dataloader)

    return total_loss, global_step


def eval_epoch(model, test_dataloader, device, args=None, epoch=0):
    """evaluation"""

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_ and is_master():
        logging.info("Eval under the multi-sentence per video clip setting.")
        logging.info("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        infer_start_t = time.time()
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            if args.save_feature_path is not None and os.path.exists(args.save_feature_path):
                if bid < 2000:
                    if args.datatype == 'msrvtt':
                        print('{}\t'.format(bid + 1), test_dataloader.dataset.data['video_id'].values[bid], end='\n')
                    if args.datatype == 'lsmdc':
                        if 'Harry_Potter' in test_dataloader.dataset.iter2video_pairs_dict[bid][0]:
                            print('{}\t'.format(bid + 1), test_dataloader.dataset.iter2video_pairs_dict[bid], end='\n')

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model(input_ids, segment_ids, input_mask)['sequence_output']
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model(video=video, video_mask=video_mask)['visual_output']
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                output = model(input_ids, segment_ids, input_mask, video, video_mask)
                batch_sequence_output_list.append(output['sequence_output'])
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(output['visual_output'])
                batch_list_v.append((video_mask,))

            if (bid + 1) % args.n_display == 0 or ( bid + 1) == len(test_dataloader):
                logging.info("{}/{}\r".format(bid, len(test_dataloader)))

        if torch.cuda.is_available(): torch.cuda.synchronize()
        all_infer_time = time.time() - infer_start_t
        logging.info('The total model inference time of the program is {:.2f} Seconds\n'.format(all_infer_time))
        if args.inference_speed_test:
            return 0

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, 
                                            batch_sequence_output_list, batch_visual_output_list, args=args)

    if multi_sentence_:
        logging.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logging.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

    else:
        logging.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logging.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))


    # return for final logging
    info_str = []
    info_str.append("Text-to-Video:")
    
    # 调试：检查tv_metrics中有哪些指标
    print(f"[DEBUG] tv_metrics keys: {list(tv_metrics.keys())}")
    print(f"[DEBUG] Available R@ metrics: {[k for k in tv_metrics.keys() if k.startswith('R')]}")
    
    # 安全地显示R1-R10的完整范围
    recall_parts = []
    for k in range(1, 11):
        key = f'R{k}'
        if key in tv_metrics:
            recall_parts.append(f"R@{k}: {tv_metrics[key]:.1f}")
        else:
            print(f"[WARNING] Missing metric: {key}")
            recall_parts.append(f"R@{k}: N/A")
    
    recall_str = " - ".join(recall_parts)
    info_str.append(f' (metric) >>> {recall_str} - Median R: {tv_metrics.get("MR", "N/A"):.1f} - Mean R: {tv_metrics.get("MeanR", "N/A"):.1f}')
    
    info_str.append("Video-to-Text:")
    
    # 调试：检查vt_metrics中有哪些指标
    print(f"[DEBUG] vt_metrics keys: {list(vt_metrics.keys())}")
    
    # 安全地显示R1-R10的完整范围
    recall_parts_vt = []
    for k in range(1, 11):
        key = f'R{k}'
        if key in vt_metrics:
            recall_parts_vt.append(f"V2T$R@{k}: {vt_metrics[key]:.1f}")
        else:
            print(f"[WARNING] Missing V2T metric: {key}")
            recall_parts_vt.append(f"V2T$R@{k}: N/A")
    
    recall_str_vt = " - ".join(recall_parts_vt)
    info_str.append(f' (metric) >>> {recall_str_vt} - V2T$Median R: {vt_metrics.get("MR", "N/A"):.1f} - V2T$Mean R: {vt_metrics.get("MeanR", "N/A"):.1f}')

    for info in info_str: logging.info(info)
    R1 = tv_metrics.get('R1', 0.0)

    return R1, all_infer_time, info_str


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list, args=None):
    """"calculate the similarity between visual output and text output"""
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model

    sim_matrix = []
    
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    # if args.camoe_dsl:
    # 	print('Apply DSL')
    # 	# https://github.com/starmemda/CAMoE
    # 	sim_matrix_ = torch.from_numpy(sim_matrix)
    # 	sim_matrix = sim_matrix_ * F.softmax(sim_matrix_, dim=0) * len(sim_matrix_)
    # 	# sim_matrix = sim_matrix_ * F.softmax(sim_matrix_, dim=1) * len(sim_matrix_)
    # 	sim_matrix = sim_matrix.cpu().numpy()

    return sim_matrix


if __name__ == "__main__":
    args = get_args()

    main(args)
    