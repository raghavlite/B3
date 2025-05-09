import logging

import PIL
from transformers.image_utils import ChannelDimension

from src.model.vlm_backbone.colpali import ColPaliProcessor

logger = logging.getLogger(__name__)

import torch
import numpy as np
from src.utils import print_master

from src.model.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.model.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.model.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from src.model.vlm_backbone.intern_vl3 import load_image
from src.model.vlm_backbone.intern_vl3 import get_conv_template
from src.model.vlm_backbone.qwen2_5_vl_tokenselection import \
    Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL_TokenSelectionForConditionalGeneration
from src.model.vlm_backbone.qwen2_vl_tokenselection import \
    Qwen2VLForConditionalGeneration as Qwen2VLTokenSelectionForConditionalGeneration, \
    Qwen2VLProcessor as Qwen2VLTokenSelectionProcessor
from src.model.vlm_backbone.internvideo2.modeling_internvideo2 import InternVideo2_Stage2
from transformers import AutoTokenizer, AutoModel
from peft import PeftConfig



PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
QWEN2_VL_TOKENSELECTION = 'qwen2_vl_tokenselection'
QWEN2_5_VL_TOKENSELECTION = 'qwen2_5_vl_tokenselection'
INTERN_VL3 = 'internvl_chat'
INTERNVIDEO2 = 'internvideo2'
GME = 'gme'  # QWEN2-VL
LamRA = 'lamra'  # QWEN2-VL
COLPALI = 'colpali'  # PaliGemma-3B
MODEL2BACKBONE = {  # keys are from hf_config.model_type or manually added if not provided
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'internvl_chat': INTERN_VL3,
    'qwen2_vl': QWEN2_VL,
    'qwen2_vl_tokenselection': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'qwen2_vl_tokenselection': QWEN2_VL_TOKENSELECTION,
    'qwen2_5_vl_tokenselection': QWEN2_5_VL_TOKENSELECTION,
    'internvideo2': INTERNVIDEO2,
    'gme': GME, 
    'lamra': LamRA,
    'colpali': COLPALI
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    INTERN_VL3: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|image_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|image_pad|>",
    GME: "<|image_pad|>",
    LamRA: "<|image_pad|>",
    INTERNVIDEO2: "",
    COLPALI: "",
}

VLM_VIDEO_TOKENS = {
    LLAVA_NEXT: "<image>",
    INTERN_VL3: "<image>",
    QWEN2_VL: "<|video_pad|>",
    QWEN2_5_VL: "<|video_pad|>",
    QWEN2_VL_TOKENSELECTION: "<|video_pad|>",
    QWEN2_5_VL_TOKENSELECTION: "<|video_pad|>",
    GME: "<|video_pad|>",
    LamRA: "<|video_pad|>",
    INTERNVIDEO2: "",
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    INTERN_VL3: AutoModel,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    QWEN2_VL_TOKENSELECTION: Qwen2VLTokenSelectionForConditionalGeneration,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_5_VL_TokenSelectionForConditionalGeneration,
    INTERNVIDEO2: InternVideo2_Stage2,
}


def load_processor(model_args, data_args=None):
    """
    Load processor based on VLM backbone.
    """
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    
    if model_args.model_backbone == PHI3V:
        from src.model.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from src.model.vlm_backbone.llava_next.processing_llava_next import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
    elif model_args.model_backbone in [QWEN2_VL, GME, LamRA]:
        from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast

        model_name_or_path = PeftConfig.from_pretrained(model_args.model_name).base_model_name_or_path if(model_args.init_lora_model) else model_name_or_path
        print(f">>>>>>>>>>>>>>>>>>>>>>>> Processor {model_name_or_path}")
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer
        )
    elif model_args.model_backbone == QWEN2_VL_TOKENSELECTION:
        from src.model.vlm_backbone.qwen2_vl_tokenselection.processing_qwen2_vl import Qwen2VLProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.do_resize = data_args.resize_use_processor
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone == QWEN2_5_VL:
        from src.model.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        from transformers import AutoProcessor
        print(f">>>>>>>>>>>>>>>>>>>>>>>> Processor 2.5 {model_name_or_path}")
        # image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path)
        # tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        # processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, image_processor=image_processor, tokenizer=tokenizer)
        # processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, image_processor=autoprocessor.image_processor, tokenizer=autoprocessor.tokenizer)
        processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side='left', max_pixels=data_args.resize_max_pixels, min_pixels = data_args.resize_min_pixels)
    elif model_args.model_backbone == INTERN_VL3:
        from transformers import AutoModel, AutoTokenizer
        print(f">>>>>>>>>>>>>>>>>>>>>>>> Tokenizer InternVL3 {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
        tokenizer.padding_side = 'left'
        class Processor:
            def __init__(self, tokenizer, num_image_token, template, img_context_token_id, system_message, padding_side, max_pixels, min_pixels):
                self.tokenizer = tokenizer
                # self.max_length = max_length
                self.num_image_token=num_image_token
                self.template=template
                self.img_context_token_id=img_context_token_id
                self.system_message=system_message
                self.padding_side = padding_side
                self.max_pixels = max_pixels
                self.min_pixels = min_pixels
            def save_pretrained(self, output_dir):
                self.tokenizer.save_pretrained(output_dir)
        
        # processor = Processor(tokenizer, num_image_token=256, template="internvl2_5", system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。', padding_side='left', max_pixels=int(np.sqrt(data_args.resize_max_pixels)), min_pixels = int(np.sqrt(data_args.resize_min_pixels)))
        processor = Processor(tokenizer, num_image_token=256, template="internvl2_5", img_context_token_id=151667, system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。', padding_side='left', max_pixels=int(np.sqrt(data_args.resize_max_pixels)), min_pixels = int(np.sqrt(data_args.resize_min_pixels)))
        # ! image processor inside the forward function in intern_vl3, num_image_tokens is the number of tokens per tile
    elif model_args.model_backbone == QWEN2_5_VL_TOKENSELECTION:
        # TODO: qwen2.5 token selection not working yet
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.model.vlm_backbone.qwen2_5_vl_tokenselection.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.model.vlm_backbone.qwen2_vl_tokenselection.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name_or_path)
        if data_args is not None:
            image_processor.do_resize = data_args.resize_use_processor
            image_processor.min_pixels = data_args.resize_min_pixels
            image_processor.max_pixels = data_args.resize_max_pixels
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_name_or_path,
            image_processor=image_processor, tokenizer=tokenizer,
            uigraph_use=model_args.uigraph_use,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone == INTERNVIDEO2:
        return None
    elif model_args.model_backbone == GME or model_args.model_backbone == LamRA:
        from src.model.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        processor = Qwen2VLProcessor.from_pretrained(model_args.model_name, min_pixels=256*28*28, max_pixels=1280*28*28)
    elif model_args.model_backbone == COLPALI:
        from transformers import AutoProcessor
        processor = ColPaliProcessor.from_pretrained(model_args.model_name)
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO: NOT FINISHED YET!
    input_ids, pixel_values, image_sizes = [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        # in theory, each batch item should contain a list of frames, but we still check for exceptions here
        # if no images as input (not likely to happen in mmeb pro cases)
        if images is None or (type(images)==list and any(i is None for i in images)):
            inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            # image_grid_thw.append(None)
        else:
            image_exists = True
            # in theory, valid images should be a list of frames
            assert isinstance(images, list), f"images should be a list, but got {type(images)}"
            inputs = processor(images=images, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask,
        'pixel_values': torch.from_numpy(np.array(pixel_values)).float(),
        'image_sizes': torch.tensor(np.array(image_sizes)).long(),
        # 'texts': texts,
        # 'images': visual_inputs,
    }

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]
    # import ipdb; ipdb.set_trace()
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            if vlm_image_token in text:
                if isinstance(images, PIL.Image.Image):
                    # images is a single image
                    images = [images]
                for iid, image in enumerate(images):
                    # rare case in MMEB eval: resize to 28*28 if either w or h is smaller than 28
                    if image.size[0] < 28 or image.size[1] < 28:
                        image = image.resize((56, 56))
                        images[iid] = image
                inputs = processor(text=[text], images=images, return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            elif vlm_video_token in text:
                # TODO: check text/video data validity
                inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(), 
        'texts': texts,
        'images': visual_inputs,
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw

    return inputs

def Gme_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    inputs = {
        'texts': model_inputs['text'],
        'images': model_inputs['images'],
    }
    return inputs


def Qwen2_VL_TokenSelection_process_fn(model_inputs: dict, processor: Qwen2VLTokenSelectionProcessor, max_length=None):
    # TODO: set separate max_len for text/visual inputs, currently max_length is only applied to text-only data
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    patch_pos, select_mask = [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, images in zip(texts, visual_inputs):
        if images is None or (type(images)==list and any(i is None for i in images)):
            # all images must be valid
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            patch_pos.append(None)
            select_mask.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            image_exists = True
            # TODO only
            # handling multi-image data from videos, cannot deal with mixed image + video data
            if VLM_IMAGE_TOKENS[QWEN2_VL] in text:
                inputs = processor(text=[text], images=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            elif VLM_VIDEO_TOKENS[QWEN2_VL] in text:
                assert len(images) > 1, f"Video data must have more than 1 frame, got {len(images)}"
                inputs = processor(text=[text], videos=[images], return_tensors="np", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError(f"Unsupported visual token in text: {text}")
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
                if 'patch_pos' in inputs:
                    patch_pos.append(inputs['patch_pos'])
                if 'select_mask' in inputs:
                    select_mask.append(inputs['select_mask'])
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                patch_pos.append(None)
                select_mask.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']

    if image_exists:
        if patch_pos:
            patch_pos_shape_for_padding = list(v.shape for v in patch_pos if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else (torch.zeros(patch_pos_shape_for_padding) - 1) for v in patch_pos]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=-1) for pos in key_tmp]
            patch_pos = torch.cat(padded_key, dim=0)
        if select_mask:
            select_mask_shape_for_padding = list(v.shape for v in select_mask if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else torch.ones(select_mask_shape_for_padding).bool() for v in select_mask]
            max_length = input_ids.size(1)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(1)), value=True) for pos in key_tmp]
            select_mask = torch.cat(padded_key, dim=0)

    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long()
    }
    inputs['pixel_values'] = pixel_values
    inputs['image_grid_thw'] = image_grid_thw
    inputs['pixel_values_videos'] = pixel_values_videos
    inputs['video_grid_thw'] = video_grid_thw
    inputs['patch_pos'] = patch_pos
    inputs['select_mask'] = select_mask

    return inputs


def InternVL_process_fn(model_inputs: dict, processor, max_length=None):
    # TODO not working yet
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['images']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def ColPali_process_fn(model_inputs: dict, processor, max_length=None):
    texts, images = model_inputs['text'], model_inputs['images']
    if images is None or all(i is None for i in images):
        inputs = processor.process_queries(texts)
    else:
        inputs = processor.process_images(images)
    return inputs


def InternVideo2_process_fn(model_inputs: dict, processor, max_length=None):
    if all(x is None for x in model_inputs["images"]):
        # Text side
        from src.model.vlm_backbone.internvideo2.modeling_internvideo2 import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        inputs = tokenizer(
            model_inputs["text"],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt")
    else:
        # Video side
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert from PIL image to tensor (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                 std=[0.229, 0.224, 0.225])  # ImageNet std
        ])
        frame_list = model_inputs["images"]
        # to make image inputs be exact 4 frames
        # Case 1: frame_list is flat (not a list of lists), e.g., [PIL, PIL, ...]
        if type(frame_list[0]) is not list:
            frame_list = [[img.copy() for _ in range(4)] for img in frame_list]
        # Case 2: frame_list is already a list of lists, ensure each has exactly 4 images
        elif type(frame_list[0]) is list and len(frame_list[0]) != 4:
            new_list = []
            for frames in frame_list:
                if len(frames) < 4:
                    frames = frames + [frames[-1].copy() for _ in range(4 - len(frames))]
                elif len(frames) > 4:
                    # Sample 4 indices uniformly across the sequence
                    indices = np.linspace(0, len(frames) - 1, num=4, dtype=int)
                    frames = [frames[i] for i in indices]
                new_list.append(frames)
            frame_list = new_list
        pixel_values = [
            torch.stack([preprocess(img) for img in frames], dim=0)  # (num_frames, C, H, W)
            for frames in frame_list
        ]

        pixel_values = torch.stack(pixel_values, dim=0)  # (B, num_frames, C, H, W)
        inputs = {'pixel_values': pixel_values}

    return inputs


def process_input_text(instruction, model_backbone, text=None, add_video_token=False, add_image_token=False):
    # Formulate input text based on text, special token and instruction.
    # TBD: Reorganize the hard-code part for internvideo2
    if model_backbone == "internvideo2":
        return text
    elif model_backbone in [GME, LamRA]:
        if text:
            return instruction + " " + text # GME and LamRA do not need special tokens
        else:
            return instruction + " "
    prompt = instruction
    if text:
        prompt = prompt + " " + text
    if add_video_token:
        video_token = VLM_VIDEO_TOKENS[model_backbone]
        prompt = video_token + " " + prompt
    if add_image_token:
        image_token = VLM_IMAGE_TOKENS[model_backbone]
        prompt = image_token + " " + prompt

    return prompt


def Intern_VL3_process_fn(model_inputs: dict, processor, max_length=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
    tokenizer=processor.tokenizer
    questions = model_inputs["text"]
    images = model_inputs["images"]

    pixel_values_list = [load_image(image[0], max_num=12).to(torch.bfloat16) if image and image[0] != '' else torch.tensor([]).reshape(0, 3, 448, 448)  for image in model_inputs["images"]]
    pixel_values = torch.cat(pixel_values_list, dim=0) 
    num_patches_list = [pv.size(0) for pv in pixel_values_list]

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    assert processor.img_context_token_id==img_context_token_id
    # import ipdb; ipdb.set_trace()

    if pixel_values is not None:
        image_bs = pixel_values.shape[0]
        # print(f'dynamic ViT batch size: {image_bs}')

    queries = []
    for idx, num_patches in enumerate(num_patches_list):
        question = questions[idx]
        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        template = get_conv_template(processor.template)
        template.system_message = processor.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * processor.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
        queries.append(query)

    
    model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
    model_inputs["pixel_values"]=pixel_values_list


    # import ipdb; ipdb.set_trace()
    return model_inputs


process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    INTERN_VL3: Intern_VL3_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
    QWEN2_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    QWEN2_5_VL_TOKENSELECTION: Qwen2_VL_TokenSelection_process_fn,
    INTERNVIDEO2: InternVideo2_process_fn,
    GME: Gme_process_fn,
    LamRA: Gme_process_fn,
    COLPALI: ColPali_process_fn,
}