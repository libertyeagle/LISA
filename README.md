# Requirements
- PyTorch 1.6.0
- numpy
- tqdm

# Usage
`train_efficient_attention_target_seperate_pq.py, train_efficient_attention_soft_pq.py, train_efficient_attention.py` correspond to the LISA-Mini, LISA-Soft, LISA-Base models, respectively.

`train_transformer.py` corresponds to the vanilla Transformer based model.

To train LISA-Mini on ML-1M dataset, run:
```
python train_efficient_attention_target_seperate_pq.py --config configs/EfficientAttn-Mini/ml-1m-256.json --train_dataset datasets/ml-1m/train.pkl --eval_dataset datasets/ml-1m/eval.pkl --gpu 0
```

To train LISA-Soft, run:
```
python train_efficient_attention_soft_pq.py --config configs/EfficientAttn-Soft/ml-1m-128.json --train_dataset datasets/ml-1m/train.pkl --eval_dataset datasets/ml-1m/eval.pkl --gpu 0
```

To train LISA-Base, run:
```
python train_efficient_attention.py --config configs/EfficientAttn-Base/ml-1m.json --train_dataset datasets/ml-1m/train.pkl --eval_dataset datasets/ml-1m/eval.pkl --gpu 0
```

To train the vanilla Transformer, run:
```
python train_transformer.py --config configs/Transformer/ml-1m.json --train_dataset datasets/ml-1m/train.pkl --eval_dataset datasets/ml-1m/eval.pkl --gpu 0
```