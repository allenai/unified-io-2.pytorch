from transformers import LlamaTokenizer

from unified_io_2 import config
from unified_io_2.get_model import get_model


def main():
  preprocessor, model = get_model(config.LARGE, "/Users/chris/data/llama_tokenizer.model")
  print(preprocessor.tokenizer)

  batch = preprocessor(
    text_inputs="Write me a story",
    image_inputs="/Users/chris/Desktop/ve.jpeg"
  )
  print(batch)


if __name__ == '__main__':
  main()