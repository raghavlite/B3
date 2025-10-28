from itertools import repeat
from typing import Optional

from torch.jit import isinstance

import logging
from dataclasses import dataclass
from transformers import ProcessorMixin, AutoProcessor, AutoTokenizer
from src.arguments import DataArguments, ModelArguments, TrainingArguments
import torch
from qwen_vl_utils import smart_resize

from src.model.processor import LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, INTERN_VL3, \
    QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, PHI3V, process_vlm_inputs_fns
from PIL import Image
import io
from src.utils import print_rank, print_master


logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000


# def process_vlm_inputs(model_inputs: dict, processor, backbone_name, max_length=None):
#     input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
#     texts, images = model_inputs['text'], model_inputs['image']
#     image_exists = False
#     # 1. iterate each pair and process (since processors do not support batch processing)
#     for text, image in zip(texts, images):
#         if image is None:
#             if backbone_name == LLAVA_NEXT:
#                 inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
#             elif backbone_name == QWEN2_VL:
#                 inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
#             elif backbone_name == PHI3V:
#                 inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
#             input_id = inputs["input_ids"].squeeze().tolist()
#             if isinstance(input_id, int):
#                 # in case of empty string, only BOS is included
#                 input_id = [input_id]
#             input_ids.append(input_id)
#             pixel_values.append(None)
#             image_sizes.append(None)
#             image_grid_thw.append(None)
#         else:
#             image_exists = True
#             if backbone_name == LLAVA_NEXT:
#                 inputs = processor(images=image, text=text, return_tensors="np", max_length=max_length, truncation=True)
#             elif backbone_name == QWEN2_VL:
#                 inputs = processor(images=[image], text=[text], return_tensors="np", max_length=max_length, truncation=True)
#             elif backbone_name == PHI3V:
#                 inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
#             input_ids.append(inputs["input_ids"].squeeze().tolist())
#             pixel_values.append(inputs['pixel_values'])
#             if 'image_sizes' in inputs:
#                 image_sizes.append(inputs['image_sizes'])
#             if 'image_grid_thw' in inputs:
#                 image_grid_thw.append(inputs['image_grid_thw'])

#     # 2. padding inputs
#     batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
#     input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
#     inputs = {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'texts': texts,
#         'images': images,
#     }
#     # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
#     if image_exists:
#         if backbone_name == LLAVA_NEXT:
#             # dummy image inputs based on the first valid data point
#             pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
#             image_size_for_padding = torch.from_numpy(list(v for v in image_sizes if v is not None)[0])
#             # make the batch full tensors
#             pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
#             pixel_values = torch.cat(pixel_values, dim=0)
#             image_sizes = [torch.from_numpy(v) if v is not None else image_size_for_padding for v in image_sizes]
#             image_sizes = torch.cat(image_sizes, dim=0)
#         if backbone_name == QWEN2_VL:
#             pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
#             pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
#             pixel_values = torch.cat(pixel_values, dim=0)
#             if image_grid_thw:
#                 image_grid_thw_for_padding = torch.from_numpy(list(v for v in image_grid_thw if v is not None)[0])
#                 image_grid_thw = [torch.from_numpy(v) if v is not None else image_grid_thw_for_padding for v in image_grid_thw]
#                 image_grid_thw = torch.cat(image_grid_thw, dim=0)
#                 inputs['image_grid_thw'] = image_grid_thw
#         # add them to inputs
#         inputs['pixel_values'] = pixel_values
#         inputs['image_sizes'] = image_sizes
#     else:
#         inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
#         inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

#     import ipdb; ipdb.set_trace()
#     # print_rank('[text.shape]' + str(input_ids.shape))
#     # if image_exists:
#     #     print_rank('[image.shape]' + str(inputs['pixel_values'].shape))

#     return inputs


# def split_dense_inputs(model_input: dict, chunk_size: int):
#     assert len(model_input) == 1
#     arg_key = list(model_input.keys())[0]
#     arg_val = model_input[arg_key]

#     keys = list(arg_val.keys())
#     chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
#     chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

#     return [{arg_key: c} for c in chunked_arg_val]




# def split_vlm_inputs(model_input: dict, chunk_size: int):
#     assert len(model_input) == 1
#     arg_key = list(model_input.keys())[0]
#     arg_val = model_input[arg_key]
#     keys = list(arg_val.keys())

#     # for input_ids and attention_mask, split directly
#     chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in ["input_ids", "attention_mask"]]

#     # for pixel_values and image_sizes, need to split based on the position of images
#     input_ids = arg_val["input_ids"]
#     # positions = torch.nonzero(((input_ids < 0) & (input_ids > -MAX_INPUT_ID)) | input_ids == LLAVE_IMAGE_TOKEN_ID, as_tuple=True)
#     positions = torch.nonzero((input_ids < 0) & (input_ids > -PHI_IMAGE_TOKEN_MAX_INPUT_ID), as_tuple=True)
#     row_contain_image = torch.unique(positions[0])  # indicates which row in input_ids contain images
#     num_chunks = len(chunked_tensors[0])
#     chunk_image_count = []
#     for chunk_idx in range(num_chunks):
#         chunk_image_count.append(torch.sum(
#             (row_contain_image >= chunk_idx * chunk_size) & (row_contain_image < (chunk_idx + 1) * chunk_size)).item())
#     if "pixel_values" in keys:
#         pixel_values = arg_val["pixel_values"]
#         image_sizes = arg_val["image_sizes"]
#         chunked_tensors.append(torch.split(pixel_values, chunk_image_count))
#         chunked_tensors.append(torch.split(image_sizes, chunk_image_count))

#     chunked_arg_val = []
#     for kk, tt in zip(repeat(keys), zip(*chunked_tensors)):
#         if "pixel_values" in keys and tt[2].numel() == 0:  # this chunk doesn't contain image
#             chunked_arg_val.append(dict(zip(kk[:2], tt[:2])))
#         else:
#             chunked_arg_val.append(dict(zip(kk, tt)))

#     return [{arg_key: c} for c in chunked_arg_val]

def split_and_process_vlm_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = []
    for k in keys:
        if isinstance(arg_val[k], torch.Tensor):
            chunked_tensor = arg_val[k].split(chunk_size, dim=0)
        else:
            chunked_tensor = [arg_val[k][i: i + chunk_size] for i in list(range(0, len(arg_val[k]), chunk_size))]
        chunked_tensors.append(chunked_tensor)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]
    chunked_inputs = [{arg_key: c} for c in chunked_arg_val]

    return chunked_inputs


def get_dense_rep(x):
    """
    Get either qry_reps or tgt_reps.
    """
    if x["qry_reps"] is None:
        return x["tgt_reps"]
    else:
        return x["qry_reps"]


@dataclass
class TrainTextImageDataCollator:
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        neg_inputs = self._get_batch_inputs(examples, "neg_text", "neg_image")
        return qry_inputs, pos_inputs

    def _get_batch_inputs(self, examples, text_keyname, image_keyname):
        texts, images = [], []
        for example in examples:
            # @ruimeng filter invalid data examples here will lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, image = '  ', None
            text, image = example[text_keyname], example[image_keyname]
            if type(text) == list:
                if len(text) == 0 or len(image) == 0:
                    text, image = '  ', None
                else:
                    text, image = text[0], image[0]
            texts.append(text)
            images.append(image)
        inputs = {'text': texts, 'image': images}
        return inputs


@dataclass
class MultimodalDataCollator:
    processor: ProcessorMixin
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    batch_size: Optional[int] = None  # used to verify if a batch has invalid data

    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        texts, visual_inputs = [], []
        for example in batch:
            # @ruimeng filter invalid data examples here may lead to fail to sync across devices (unequal batch size)
            # use dummy input for now
            if example is None or not example:
                text, visual_input = '  ', None
                texts.append(text)
                visual_inputs.append(visual_input)
                assert False, "This should not happen"
            else:
                text_list, raw_images_list = example[text_keyname], example[image_keyname]
                for text,raw_images in zip(text_list, raw_images_list):
                    if type(raw_images) == dict:
                        visual_input = []
                        assert 'resolutions' in raw_images, "we need len(raw_images['resolutions']) to determine the number of images, set it a list of None of for cases that no resizing is needed"
                        num_images = len(raw_images['resolutions'])
                        for image_idx in range(num_images):

                            bytes = raw_images['bytes'][image_idx] if 'bytes' in raw_images else None
                            path = raw_images['paths'][image_idx] if 'paths' in raw_images else None
                            image_resolution = raw_images['resolutions'][image_idx] if 'resolutions' in raw_images else None
                            if self.model_args.model_backbone in [INTERN_VL3]:
                                image=path
                            else:
                                if bytes is None and ((path is None) or (not path)):
                                    image = None
                                elif bytes is not None:
                                    # vidore, image inputs are already bytes
                                    image = Image.open(io.BytesIO(bytes))
                                elif (path is not None) and (path):
                                    # mmeb/video datasets, lazy image loading and processing
                                    image = Image.open(path)
                                else:
                                    print_rank(f"\n{'=' * 50}\nsomething went wrong with a data point from {example['global_dataset_name']}, neither bytes or path is given. \n\t\tquery_text: {example['query_text']}")
                                if not self.data_args.resize_use_processor and image is not None and image_resolution:
                                    image = image.resize(image_resolution)
                                # if image is not None and self.data_args.image_decay_factor < 1.0:
                                #     assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                                #     assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                                #     # TODO: this is a hacky way to decay image resolution, need to be refactored
                                #     max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))
                                if image is not None and False:
                                    assert False
                                    if self.data_args.image_decay_factor is None:
                                        max_pixels = self.data_args.resize_max_pixels
                                    else:
                                        max_pixels = max(self.data_args.resize_min_pixels, self.data_args.resize_max_pixels * self.data_args.image_decay_factor ** (num_images - image_idx))                            

                                    assert image_resolution is None, "image_resolution is conflicting with image_decay_factor"
                                    assert self.model_args.model_backbone in [QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION], "image_decay_factor is only supported for Qwen models"
                                    # import ipdb; ipdb.set_trace()
                                    width, height = image.size
                                    if(width*height > max_pixels):
                                        resized_height, resized_width = smart_resize(
                                            height,
                                            width,
                                            min_pixels=self.data_args.resize_min_pixels,
                                            max_pixels=max_pixels,
                                        )
                                        print(">>>>>>>>>>>>>", (width, height), (resized_width, resized_height), flush=True)
                                        image = image.resize((resized_width, resized_height))
                                    else:
                                        pass
                            visual_input.append(image)
                    else:
                        visual_input = None
                    texts.append(text)
                    visual_inputs.append(visual_input)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs


    def __call__(self, examples):
        """
        :param examples: 'query_text', 'query_image_path', 'pos_text', 'pos_image_path', 'neg_text', 'neg_image_path'
        """
        # print_rank(str([e['global_dataset_name'] for e in examples]))
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")
        # neg_inputs = self._get_batch_inputs(examples, "neg_text", "neg_image")
        bs = len(qry_inputs['text'])
        assert bs > 0, 'An empty batch'
        # pad batch to batch_size to avoid hanging in distributed training
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expect batch size {self.batch_size}, but got batch size of {bs}")
            pass
        process_fn = process_vlm_inputs_fns[self.training_args.model_backbone]
        processed_qry_inputs = process_fn(qry_inputs, processor=self.processor, max_length=self.data_args.max_len)
        processed_pos_inputs = process_fn(pos_inputs, processor=self.processor, max_length=self.data_args.max_len)
        # print_rank(f"\t\tQry collator: processed_qry_inputs['input_ids'].shape={processed_qry_inputs['input_ids'].shape}")
        # print_rank(f"\t\tPos collator: processed_pos_inputs['input_ids'].shape={processed_pos_inputs['input_ids'].shape}")
        if('input_ids' in processed_qry_inputs):
            print_rank(f"\t\tQry collator: processed_qry_inputs['input_ids'].shape={processed_qry_inputs['input_ids'].shape}\t\tPos collator: processed_pos_inputs['input_ids'].shape={processed_pos_inputs['input_ids'].shape}")
        processed_qry_inputs['text'] = [f for e in examples for f in e['query_text']]
        processed_pos_inputs['text'] = [f for e in examples for f in e['pos_text']]
        processed_qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        processed_pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples for f in e['pos_text']]
    
        # print_rank(f"{processed_pos_inputs['text'][:5]}; {processed_pos_inputs['global_dataset_name'][0]}; {[example['idx'] for example in examples]}" )
        print_rank(f"::::::::::::::::::::::{[example['idx'] for example in examples]}" )
        return processed_qry_inputs, processed_pos_inputs
