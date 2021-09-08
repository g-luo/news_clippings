import torch
import numpy as np
from typing import Dict
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from news_clippings_training.datasets.database import NewsCLIPpingsAnnotationDatabase

class NewsCLIPpingsDataset(MMFDataset):
	def __init__(self, config: Dict, dataset_type: str, index, *args, **kwargs):
		super().__init__("news_clippings", config, dataset_type, index, *args, **kwargs)
		# TODO(g-luo): np is currently unused
		np.random.seed(2)

	# Example annotations formatting
	# {"annotations": [sample_1, sample_2, ...]}
	def build_annotation_db(self):
		"""
		Override build_image_db and build_feature_db for to implement custom versions of
		those for a particular dataset.
		"""
		annotation_path = self._get_path_based_on_index(
			self.config, "annotations", self._index
		)
		return NewsCLIPpingsAnnotationDatabase(self.config, annotation_path)

	def init_processors(self):
		super().init_processors()
		if self._use_images and hasattr(self, "image_processor"):
			self.image_db.transform = self.image_processor

	def __getitem__(self, idx: int):
		# ============================================
		# 			   Tokenize caption
		# ============================================
		sample_info = self.annotation_db[idx]
		return self.create_item(sample_info, idx)

	def create_item(self, sample_info, idx):
		# Example sample formatting
		# {
		#    id: 123
		#    image_id: 456
		#    caption: "this is a caption"
		#    falsified: True
		# }
		current_sample = Sample()

		processed_caption = self.text_processor({"text": sample_info["caption"]})
		current_sample.update(processed_caption)
		current_sample.caption = sample_info["caption"]

		if "id" in sample_info:
			sample_id = sample_info["id"]
		else:
			sample_id = idx
			
		current_sample.id = torch.tensor(
			int(sample_id), dtype=torch.int
		)

		if self._use_images:
			current_sample["image_path"] = self.get_image_path(sample_info)
			current_sample.image = self.image_db.from_path(current_sample["image_path"])["images"][0]

		# TODO(g-luo): Create a script to hydrate the data.
		# target = sample_info["falsified"]
		target = sample_info["foil"]
		current_sample.targets = torch.tensor(int(target), dtype=torch.long)
		
		return current_sample

	def get_image_path(self, sample_info):
		"""
		Please note that only jpg images are currently supported.
		"""
		image_path = sample_info["image_path"]
		return image_path

	def get_feature_path(self, image_path):
		# TODO(g-luo): Implement this for VisualBERT.
		pass

	# ================================================
	#         Formatting used by mmf_predict
	# ================================================
	def format_for_prediction(self, report):
		output = []
		for idx, id in enumerate(report.id):
			target = report.targets[idx].item()
			score = report.scores[idx].argmax().item()
			logits = report.scores[idx].detach().cpu().tolist()
			caption = report.caption[idx]
			output.append({
											"id": id.item(), 
											"target": target,
											"score": score,
											"logits": logits,
											"caption": caption
										})
		return output
