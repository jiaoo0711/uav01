# coding=utf-8
"""dataset for MSRVTT
"""
from __future__ import absolute_import, division, unicode_literals

import os
import json
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from .decode import RawVideoExtractorpyAV
import logging


class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            lmdb_dataset=None
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.tokenizer = tokenizer
        self.effective_max_words = self.tokenizer.model_max_length
        logging.info(f"[MSRVTT_DataLoader] Effective max_words for tokenization: {self.effective_max_words} (from tokenizer.model_max_length)")
        
        self.max_frames = max_frames
        self.lmdb_dataset = lmdb_dataset

        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution, is_train=False,
                                                        num_segments=self.max_frames,
                                                        lmdb_dataset=self.lmdb_dataset)

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        k = len(choice_video_ids)

        current_max_length = self.effective_max_words

        all_input_ids = np.zeros((k, current_max_length), dtype=np.int64)
        all_attention_mask = np.zeros((k, current_max_length), dtype=np.int64)
        all_segment_ids = np.zeros((k, current_max_length), dtype=np.int64)

        for i, vid_id in enumerate(choice_video_ids):
            encoding = self.tokenizer(
                [sentence],
                padding='max_length',
                truncation=True,
                max_length=current_max_length,
                return_tensors="np",
                return_token_type_ids=True
            )
            
            input_ids = encoding['input_ids'][0]
            attention_mask = encoding['attention_mask'][0]
            
            all_input_ids[i] = input_ids
            all_attention_mask[i] = attention_mask

            if 'token_type_ids' in encoding and encoding['token_type_ids'] is not None:
                 all_segment_ids[i] = encoding['token_type_ids'][0]

        return all_input_ids, all_attention_mask, all_segment_ids, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)
        video_list = []
        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if not isinstance(video_path, str):
                video_path = video_path.decode('utf-8')
            raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(video_path)
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
            video_list.append(raw_video_data)
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
        video = torch.stack(video_list, dim=0)

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        all_input_ids, all_attention_mask, all_segment_ids, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return all_input_ids, all_attention_mask, all_segment_ids, video, video_mask


class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            lmdb_dataset=None
    ):
        """
        MSRVTT training dataset.
        MSRVTT has 200000 sentences, 10000 videos.
        =============================
        Args:
            csv_path: path to the video list
            json_path: text corpus
            features_path: video path
            tokenizer: text tokenizer
            feature_framerate: fps
        """
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        
        self.tokenizer = tokenizer
        self.effective_max_words = self.tokenizer.model_max_length
        logging.info(f"[MSRVTT_TrainDataLoader] Effective max_words for tokenization: {self.effective_max_words} (from tokenizer.model_max_length)")

        self.max_frames = max_frames
        self.lmdb_dataset = lmdb_dataset
        
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])
            self.sample_len = len(self.csv)

        logging.info(f"Train Dataloader: Unfold sentences: {self.unfold_sentences}, Sample length: {self.sample_len}")

        self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution, is_train=True, \
                                                        num_segments=self.max_frames, \
                                                        lmdb_dataset=self.lmdb_dataset)
        
        # Flag to enable FG-CLIP long/short text generation
        # This will be set by the dataloader factory based on args.use_fgclip
        self.use_fgclip_long_short = False

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None, max_length=None):
        """
        Get text encoding for a caption.
        Args:
            video_id: video ID
            caption: caption text (if None, randomly select from video's captions)
            max_length: maximum token length (if None, use self.effective_max_words)
        """
        if max_length is None:
            current_max_length = self.effective_max_words
        else:
            current_max_length = max_length
        
        k = 1
        if caption is None:
            captions_for_video = self.sentences[video_id]
            if not captions_for_video:
                logging.warning(f"No captions found for video_id: {video_id}")
                caption_text = " "
            else:
                caption_text = random.choice(captions_for_video)
        else:
            caption_text = caption

        encoding = self.tokenizer(
            [caption_text],
            padding='max_length',
            truncation=True,
            max_length=current_max_length,
            return_tensors="np",
            return_token_type_ids=True
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        segment_ids = np.zeros_like(input_ids, dtype=np.int64)
        if 'token_type_ids' in encoding and encoding['token_type_ids'] is not None:
            segment_ids = encoding['token_type_ids']
        
        return input_ids, attention_mask, segment_ids
    
    def _generate_short_text(self, long_caption):
        """
        Generate short text from long caption for FG-CLIP.
        Strategy: Add "a photo of" prefix and truncate to fit 77 tokens.
        """
        # Simple strategy: add prefix and truncate
        # More sophisticated strategies could be used (e.g., extract key phrases)
        short_caption = "a photo of " + long_caption
        # The tokenizer will handle truncation to 77 tokens
        return short_caption

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)
        video_list = []
        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if not isinstance(video_path, str):
                video_path = video_path.decode('utf-8')
            raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(video_path)
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
            video_list.append(raw_video_data)
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
        video = torch.stack(video_list, dim=0)

        return video, video_mask

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        
        # Get the actual caption text for generating short text
        if caption is None:
            captions_for_video = self.sentences[video_id]
            if not captions_for_video:
                caption_text = " "
            else:
                caption_text = random.choice(captions_for_video)
        else:
            caption_text = caption
        
        # Check if we need to generate long and short text for FG-CLIP
        use_fgclip_long_short = getattr(self, 'use_fgclip_long_short', False)
        
        if use_fgclip_long_short:
            # Generate long text (248 tokens) and short text (77 tokens)
            # Long text: use the full caption with max_length=248
            input_ids_long, attention_mask_long, segment_ids_long = self._get_text(
                video_id, caption_text, max_length=248
            )
            
            # Short text: generate from long caption with max_length=77
            short_caption = self._generate_short_text(caption_text)
            input_ids_short, attention_mask_short, segment_ids_short = self._get_text(
                video_id, short_caption, max_length=77
            )
            
            video, video_mask = self._get_rawvideo([video_id])
            return (input_ids_long, attention_mask_long, segment_ids_long,
                    input_ids_short, attention_mask_short, segment_ids_short,
                    video, video_mask)
        else:
            # Original behavior: single text input
            input_ids, attention_mask, segment_ids = self._get_text(video_id, caption_text)
            video, video_mask = self._get_rawvideo([video_id])
            return input_ids, attention_mask, segment_ids, video, video_mask
