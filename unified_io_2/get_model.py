from typing import Tuple

from unified_io_2.config import Config
from unified_io_2.get_modality_processor import get_input_modalities, get_target_modalities
from unified_io_2.preprocessing import UnifiedIOPreprocessing
from unified_io_2.model import UnifiedIOModel


def get_model(config: Config, tokenizer_path) -> Tuple[UnifiedIOPreprocessing, UnifiedIOModel]:
  """Return a model (with new initialized parameters) and preprocess for the configuration"""
  preprocessor = UnifiedIOPreprocessing.from_config(config, tokenizer_path)
  model = UnifiedIOModel(config)
  return preprocessor, model
