# Unified-IO-2.PyTorch

This repo is pytorch implementation of unified-io-2, which can used for inference evaluation and finetuning. 

Install

```
git clone unified-io-2.pytorch
cd unified-io-2.pytorch
```

```
pip install -r requirements.txt
```

(Optional) Use Flash Attention 2 (only available in PyTorch 2.2)
```
pip uninstall -y torch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

## Use the model

copy the pretrained model to local. 
```
gsutil -m cp -r gs://unified-io-2-us-east/checkpoints/unified-io-2_large_instructional_tunning_2M.npz ./checkpoints
```


## Finetune the model
```
python finetune/full.py
```


