from typing import Tuple

from uio2.config import Config
from uio2.get_modality_processor import get_input_modalities, get_target_modalities
from uio2.preprocessing import UnifiedIOPreprocessor
from uio2.model import UnifiedIOModel


def get_model(config: Config, tokenizer_path) -> Tuple[UnifiedIOPreprocessor, UnifiedIOModel]:
  """Return a model (with new initialized parameters) and preprocess for the configuration"""
  preprocessor = UnifiedIOPreprocessor.from_config(config, tokenizer_path)
  model = UnifiedIOModel(config)
  return preprocessor, model
