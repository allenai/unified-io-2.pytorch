# UnifiedIO-2.PyTorch

This repo is an official pytorch port of [UnifiedIO-2](https://unified-io-2.allenai.org/). The original jax code can be found
[here](https://github.com/allenai/unified-io-2). UnifiedIO 2 is a multi-modal multi-task model capable of performing a wide
range of tasks.

![test](teaser-short.svg)

## Installation
Install [pytorch](https://pytorch.org/) following the recommendation for your system. Then install with

```
git clone unified-io-2.pytorch
cd unified-io-2.pytorch
pip install -r requirements.txt
```

## Loading the model

Load the model with 
```
from uio2.model import UnifiedIOModel
model = UnifiedIOModel.from_pretrained("allenai/uio2-large")
```
This loads the large (1B) model, load the XL (3B) or XXL (7B) with 
`chrisc36/uio2-dbg-xl` and `chrisc36/uio2-dbg-xxl`.

This model requires pre-processed tensor inputs. Pre-processing is done by `UnifiedIOPreprocessing`: 

```
from uio2.preprocessing import UnifiedIOPreprocessing 
preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor", tokenizer="/path/to/tokenizer")
```

Here "/path/to/tokenizer" needs to point to the LLaMa tokenizer file. The tokenizer
file needs to be downloaded manually from [LLaMA](https://llama.meta.com/).

You can remove modality-specific components you don't need. For example,
if you only want to do text-to-image tasks run:

```
model.set_modalities(input_modalities=["text"], target_modalities=["image"])
```


This will remove some unneeded parameters from the model.

### Initializing from Scratch
The model can also be built from scratch by directly using a config:

```
from uio2 import config 
preprocessor = UnifiedIOPreprocessing.from_config(config.LARGE, /path/to/tokenizer)
model = UnifiedIO(config.LARGE)
```

### Using bfloat16
The model can be run in `bfloat16`, typically we have done this while keeping the ViTs
 and VQGANs as `float32`. To convert the model to this format run:
```
model.to_dtype(torch.bfloat16, vit_dtype=torch.float32, vqgan_dtype=torch.float32)
```

We provide pre-trained models in this format to reduce memory/bandwidth requirements 
when downloading/loading the models:  

```
model = UnifiedIOModel.from_pretrained("allenai/uio2-large-bfloat16")
```

## Usage
### Generation
Do text generation

```
from uio2.preprocessing import build_batch 
preprocessed_example = preprocessor(text_inputs="What color is the sky?", target_modality="text")
batch = build_batch([preprocessed_example], device=model.device)
tokens = model.generate(batch, modality="text", max_new_tokens=128)
```

`modality` can be set to `"image"` or `"audio"`. Image will return a `[256, 256, 3]` image, and 
audio will return a `[128. 256, 1]` mel-spectrogram. 

To see many other examples of generation and how to best configure the model and post-process
the output, see `TaskRunner` 

```
from uio2.runner import TaskRunner

runner = TaskRunner(model, preprocessor)
image = runner.image_generation("a cat")
wavform = runner.audio_generation("dogs barking")
box = runner.refexp("/path/to/image", "the green car")
keypoint = runner.keypoint("/path/to/image")
# And many more, see TaskRunner
```

### Answer Scoring
`model.score_answer_options` can compute the loss of several possible
outputs given one set of inputs. See `TaskRunner.categorization` or `TaskRunner.box_categorization` to see 
examples of how to use it.  

```
runner.categorization("/path/to/image", ["cat", "dog"])
```


### Computing the Loss
Calling the model will produce logits, masks, and targets for each modality.
If using forward, at least one target modality should be set when calling the 
preprocessor.

The loss for an example can then be computed like this:

```
from torch.nn import functional as F
from uio2.preprocessing import build_batch
preprocessed_example = preprocessor(
text_inputs="What is 1+1?", text_targets="2", target_modality="text")
batch = build_batch([preprocessed_example], device=model.device)
out = model(batch)
total_loss = 0
for modality, (logits, targets, mask) in out.items():
    losses = F.cross_entropy(
      logits.view(-1, logits.shape[-1]), targets.view(-1).to(torch.long), reduction="none")
    total_loss += (losses.reshape(logits.shape[:2])*mask)/mask.sum()
print(total_loss)
```

See `preprocessor` supports inputs/output for all modalities. 

To train the model, run `preprocessor` and `build_batch` in a DataLoader and then
backprop on the loss. 

## Citation

```bibtex
@article{lu2023uio2,
  title   = {Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action}, 
  author  = {Jiasen Lu and Christopher Clark and Sangho Lee and Zichen Zhang and Savya Khosla and Ryan Marten and Derek Hoiem and Aniruddha Kembhavi},
  journal = {arXiv preprint arXiv:2312.17172},
  year    = {2023},
}
```
