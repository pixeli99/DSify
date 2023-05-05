import os
import random
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

from glob import glob
import torch

import InternVideo

class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }


class CCSBUAlignDataset(CaptionDataset):

    def __getitem__(self, index):
        # Modified to accommodate the reading of BDDX data
        dummy_id = 0
        ann = self.annotation[index]

        video_file = ann["event_path"]
        image_tensors = InternVideo.load_images(os.path.join(self.vis_root, video_file,))
        caption = ann["caption"]

        return {
            "image": image_tensors,
            "text_input": caption,
            "image_id": dummy_id,
        }
    
class COCODetailDataset(CaptionDataset):

    def __getitem__(self, index):
        dummy_id = 0
        ann = self.annotation[index]

        # img_file = '{}.jpg'.format(ann["image_id"])
        image_path  = ann["image"]
        image_path = os.path.join(self.vis_root, image_path )
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        # Randomly select a conversation from the list
        convo_len = len(ann['conversations'])
        convo_idx = random.randint(0, convo_len - 2) // 2 * 2

        # Ensure that the selected conversation is from a human
        assert ann['conversations'][convo_idx]['from'] == 'human'
        ask, answer = ann['conversations'][convo_idx]['value'].strip('<image>').replace("\n", ""), ann['conversations'][convo_idx + 1]['value'].strip('<image>').replace("\n", "")

        return {
            "image": image,
            "text_prompt": "###Human: <Media><MediaHere></Media> {} ###Assistant: ".format(ask),
            "text_input": answer,
            "image_id": dummy_id,
        }