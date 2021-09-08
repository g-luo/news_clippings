from mmf.datasets.processors.processors import BaseProcessor
from mmf.common.registry import registry
import torch
from news_clippings_training.models.CLIP.clip.simple_tokenizer import SimpleTokenizer, default_bpe
		
@registry.register_processor("clip_tokenizer")
class CLIPTokenizer(BaseProcessor):
	def __init__(self, config, *args, **kwargs):
		super().__init__(config, *args, **kwargs)
		tokenizer_path = config.get("tokenizer_path", default_bpe())
		self.tokenizer = SimpleTokenizer(bpe_path=tokenizer_path)
		self.sot_token = self.tokenizer.encoder['<|startoftext|>']
		self.eot_token = self.tokenizer.encoder['<|endoftext|>']

		self.max_seq_length = config.get("max_seq_length", 77)

	def __call__(self, item):
		processed_item = {}
		# Take 77 tokens
		input_ids, tokens = self.tokenizer.encode(item["text"])
		input_ids = input_ids[:self.max_seq_length-2]
		tokens = tokens[:self.max_seq_length-2]
		input_ids = [self.sot_token] + input_ids + [self.eot_token]
		input_padding = torch.zeros(self.max_seq_length - len(input_ids))

		# Tokens may not be the same length as input_ids
		processed_item["tokens"] = tokens
		processed_item["input_ids"] = torch.cat((torch.Tensor(input_ids), input_padding)).long()
		# processed_item["input_ids"].to(self.device)
		processed_item["lm_label_ids"] = -1 * torch.ones((self.max_seq_length, 1))
		processed_item["segment_ids"] = torch.zeros((self.max_seq_length, 1))
		processed_item["input_mask"] = processed_item["input_ids"] != 0
		return processed_item