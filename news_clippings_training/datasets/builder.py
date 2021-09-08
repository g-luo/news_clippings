from mmf.common.registry import registry
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from news_clippings_training.datasets.news_clippings import NewsCLIPpingsDataset

@registry.register_builder("news_clippings")
class NewsCLIPpingsBuilder(MMFDatasetBuilder):
		def __init__(
			self, 
			dataset_name="news_clippings", 
			dataset_class=NewsCLIPpingsDataset,
			*args, 
			**kwargs,
		):
			super().__init__(dataset_name, dataset_class, *args, **kwargs)
	
		@classmethod
		def config_path(cls):
			return "configs/news_clippings.yaml"