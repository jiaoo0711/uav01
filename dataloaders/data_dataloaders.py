# coding=utf-8
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
import logging

def dataloader_msrvtt_train(args, tokenizer):
    # Determine max_words for the dataloader based on FG-CLIP usage
    if args.use_fgclip:
        # When using FG-CLIP, main.py sets tokenizer.model_max_length from args.fg_clip_max_len.
        # The dataloader should respect this length.
        max_words_for_dataloader = tokenizer.model_max_length
        logging.info(f"[dataloader_msrvtt_train] Using FG-CLIP. max_words for DataLoader set to tokenizer.model_max_length: {max_words_for_dataloader}")
    else:
        # Original logic for non-FGCLIP or if max_words_long was intended for other CLIP versions
        max_words_for_dataloader = args.max_words 
        logging.info(f"[dataloader_msrvtt_train] Not using FG-CLIP or specific FG-CLIP length not set. max_words for DataLoader set to args.max_words: {max_words_for_dataloader}")

    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=max_words_for_dataloader,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    
    # Enable FG-CLIP long/short text generation if using FG-CLIP
    if args.use_fgclip:
        msrvtt_dataset.use_fgclip_long_short = True
        logging.info("[dataloader_msrvtt_train] Enabled FG-CLIP long/short text generation")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    # Determine max_words for the dataloader based on FG-CLIP usage
    if args.use_fgclip:
        # When using FG-CLIP, main.py sets tokenizer.model_max_length from args.fg_clip_max_len.
        # The dataloader should respect this length.
        max_words_for_dataloader = tokenizer.model_max_length
        logging.info(f"[dataloader_msrvtt_test] Using FG-CLIP. max_words for DataLoader set to tokenizer.model_max_length: {max_words_for_dataloader}")
    else:
        # Original logic for non-FGCLIP
        max_words_for_dataloader = args.max_words
        logging.info(f"[dataloader_msrvtt_test] Not using FG-CLIP. max_words for DataLoader set to args.max_words: {max_words_for_dataloader}")

    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=max_words_for_dataloader,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test, "test":None}
