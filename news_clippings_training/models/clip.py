import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from typing import Optional, Dict
from torch import nn, Tensor
from mmf.modules.layers import MLPClassifer
import news_clippings_training.models.CLIP.clip as clip
from news_clippings_training.models.utils import get_optimizer_parameters_custom, freeze_optimizer_parameters_clip

EMBEDDING_DIMS = {"RN101": (512, 512), "ViT-B/32": (512, 512), "RN50": (1024, 1024)}
REMAPPING = {"vitb32": "ViT-B/32", "rn101": "RN101", "rn50": "RN50"}

@registry.register_model("clip")
class CLIP(BaseModel):

	def build(self):
		clip_model_type = self.config.get("clip_model_type")
		if clip_model_type not in REMAPPING.values():
			clip_model_type = REMAPPING[clip_model_type]
		self.model, _ = clip.load(clip_model_type, device="cuda", jit=False)
		print(f"Using CLIP model type {clip_model_type}")

		# Do this since original model is fp16
		self.model = self.model.float()
		self.model.config = self.config

		freeze_optimizer_parameters_clip(self.config, self.model)
		
		text_embedding_dim, image_embedding_dim = EMBEDDING_DIMS[clip_model_type]
		hidden_size = self.config.get("text_embedding_dim", text_embedding_dim) + self.config.get("image_embedding_dim", image_embedding_dim)
		self.model.classifier = MLPClassifer (
				hidden_size, 
				self.config.num_labels,
				hidden_dim=self.config.mlp_hidden_dim,
				num_layers=self.config.mlp_num_layers,
				dropout=self.config.mlp_dropout,
				hidden_act=self.config.mlp_act,
				batch_norm=False
		)
		self.text_only = self.config.get("text_only", False)
		self.image_only = self.config.get("image_only", False)
		print("Text only: ", self.text_only)
		print("Image only: ", self.image_only)
	
	def get_optimizer_parameters(self, config):
		return get_optimizer_parameters_custom(config, self.model)

	def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
		output_dict = {}
		
		input_ids = sample_list["input_ids"]
		image = sample_list["image"]

		# Zero out a modality if the model should be unimodal
		if self.text_only:
			image = torch.zeros(image.shape, device="cuda").long()
		elif self.image_only:
			input_ids = torch.zeros(input_ids.shape, device="cuda").long()

		text_features = self.model.encode_text(input_ids).float()
		image_features  = self.model.encode_image(image).float()
		
		text_features = text_features / text_features.norm(dim=-1, keepdim=True)
		image_features = image_features / image_features.norm(dim=-1, keepdim=True)

		text_features.to("cuda")
		image_features.to("cuda")

		features = torch.cat((text_features, image_features), dim=1)
		logits = self.model.classifier(features)
		reshaped_logits = logits.contiguous().view(-1, self.config.num_labels)

		output_dict["scores"] = reshaped_logits
		return output_dict
