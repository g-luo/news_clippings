import json
from mmf.datasets.databases.annotation_database import AnnotationDatabase
import logging

logger = logging.getLogger(__name__)

class NewsCLIPpingsAnnotationDatabase(AnnotationDatabase):

  def load_annotation_db(self, path: str):
    self.data = json.load(open(path))['annotations']

  def __len__(self):
    return self.get_len()

  def get_len(self):
    return len(self.data) - self.start_idx