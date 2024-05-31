# Lipsum-FT

## BibTeX

```
@inproceedings{nam2024lipsumft,
  title     = {Lipsum-{FT}: Robust Fine-Tuning of Zero-Shot Models Using Random Text Guidance},
  author    = {Giung Nam and Byeongho Heo and Juho Lee},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=2JF8mJRJ7M},
}
```

## Prepare training datasets

It uses TFDS datasets for training:
```
~/tensorflow_datasets/domainnet/real/1.0.0/*
~/tensorflow_datasets/imagenet2012/5.1.0/*
```

It uses preprocessed numpy arrays for evaluation:
```
/mnt/local/datasets/manual/DomainNetClipart_x224/*
/mnt/local/datasets/manual/DomainNetInfograph_x224/*
/mnt/local/datasets/manual/DomainNetPainting_x224/*
/mnt/local/datasets/manual/DomainNetReal_x224/*
/mnt/local/datasets/manual/DomainNetSketch_x224/*
/mnt/local/datasets/manual/ImageNetA_x224/*
/mnt/local/datasets/manual/ImageNetR_x224/*
/mnt/local/datasets/manual/ImageNetSketch_x224/*
/mnt/local/datasets/manual/ImageNetV2_x224/*
/mnt/local/datasets/manual/ImageNet_x224/*
```

Download links for the preprocessed numpy arrays:
```
https://www.dropbox.com/s/037tnnjbneze1na/DomainNetClipart_x224.tar.gz
https://www.dropbox.com/s/gbtkn8kd8m0bggb/DomainNetInfograph_x224.tar.gz
https://www.dropbox.com/s/gvhmxlaebvcy8zg/DomainNetPainting_x224.tar.gz
https://www.dropbox.com/s/uenqdkied5kqzfz/DomainNetReal_x224.tar.gz
https://www.dropbox.com/s/wec7qze8o9lo4r8/DomainNetSketch_x224.tar.gz
https://www.dropbox.com/s/gz5ac5xcv4hnkcw/ImageNet_x224.tar.gz
https://www.dropbox.com/s/qxathc2to6oqp51/ImageNetA_x224.tar.gz
https://www.dropbox.com/s/9bvezn0vcu2efk8/ImageNetR_x224.tar.gz
https://www.dropbox.com/s/wl8vmh8ocfejuxi/ImageNetSketch_x224.tar.gz
https://www.dropbox.com/s/4npypel3gatmisv/ImageNetV2_x224.tar.gz
```

## Prepare zero-shot heads 

### DomainNet

Command lines:
```
python scripts/setup.py --data_name domainnet --evaluate True
```

Console outputs:
```
(...)
[2024-04-24 13:40:41] Create zeroshot weights for domainnet...
[2024-04-24 13:40:59] zeroshot_raw_weights.shape: (512, 80, 345)
[2024-04-24 13:40:59] zeroshot_cls_weights.shape: (512, 345)
[2024-04-24 13:40:59] zeroshot_ctx_weights.shape: (512, 80)
[2024-04-24 13:43:27] DomainNetReal/acc 8.246e-01, DomainNetReal/nll 7.305e-01, DomainNetPainting/acc 6.669e-01, DomainNetPainting/nll 1.498e+00, DomainNetClipart/acc 6.557e-01, DomainNetClipart/nll 1.524e+00, DomainNetInfograph/acc 4.377e-01, DomainNetInfograph/nll 3.072e+00, DomainNetSketch/acc 5.698e-01, DomainNetSketch/nll 2.046e+00
```

Output files:
```
save/clip/openai/clip-vit-base-patch16/logit_scale.ckpt
save/clip/openai/clip-vit-base-patch16/text_projection.ckpt
save/clip/openai/clip-vit-base-patch16/visual_projection.ckpt
save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/domainnet.ckpt
```

### ImageNet

Command lines:
```
python scripts/setup.py --data_name imagenet2012 --evaluate True
```

Console outputs:
```
(...)
[2024-04-24 16:52:22] Create zeroshot weights for imagenet2012...
[2024-04-24 16:53:04] zeroshot_raw_weights.shape: (512, 80, 1000)
[2024-04-24 16:53:04] zeroshot_cls_weights.shape: (512, 1000)
[2024-04-24 16:53:04] zeroshot_ctx_weights.shape: (512, 80)
[2024-04-24 16:56:05] ImageNet/acc 6.817e-01, ImageNet/nll 1.184e+00, ImageNetV2/acc 6.198e-01, ImageNetV2/nll 1.490e+00, ImageNetR/acc 7.635e-01, ImageNetR/nll 9.267e-01, ImageNetA/acc 5.200e-01, ImageNetA/nll 1.924e+00, ImageNetSketch/acc 4.666e-01, ImageNetSketch/nll 2.245e+00
```

Output files:
```
save/clip/openai/clip-vit-base-patch16/logit_scale.ckpt
save/clip/openai/clip-vit-base-patch16/text_projection.ckpt
save/clip/openai/clip-vit-base-patch16/visual_projection.ckpt
save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/imagenet2012.ckpt
```

## Start training runs

### DomainNet

Command lines:
```
python scripts/Lipsum-FT.py --clip_cls_init save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/domainnet.ckpt --data_name domainnet --optim_ni 5000 --optim_lr 1e-05 --seed 0 --save save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/domainnet/Lipsum-FT/0/
```

Console outputs:
```
(...)
[2024-04-24 16:33:04] DomainNetReal/acc 8.890e-01, DomainNetReal/nll 4.246e-01, DomainNetPainting/acc 6.638e-01, DomainNetPainting/nll 1.519e+00, DomainNetClipart/acc 6.808e-01, DomainNetClipart/nll 1.389e+00, DomainNetInfograph/acc 4.629e-01, DomainNetInfograph/nll 3.061e+00, DomainNetSketch/acc 5.657e-01, DomainNetSketch/nll 2.143e+00
```

Output files:
```
save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/domainnet/Lipsum-FT/0/best_acc.ckpt
save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/domainnet/Lipsum-FT/0/*.log
```

### ImageNet

Command lines:
```
python scripts/Lipsum-FT.py --clip_cls_init save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/imagenet2012.ckpt --data_name imagenet2012 --optim_ni 50000 --optim_lr 1e-05 --seed 0 --save save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/imagenet2012/Lipsum-FT/0/
```

Console outputs:
```
(...)
[2024-04-24 22:51:55] ImageNet/acc 8.332e-01, ImageNet/nll 5.935e-01, ImageNetV2/acc 7.356e-01, ImageNetV2/nll 1.011e+00, ImageNetR/acc 7.586e-01, ImageNetR/nll 9.702e-01, ImageNetA/acc 4.987e-01, ImageNetA/nll 1.988e+00, ImageNetSketch/acc 5.148e-01, ImageNetSketch/nll 2.127e+00
```

Output files:
```
save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/imagenet2012/Lipsum-FT/0/best_acc.ckpt
save/clip/openai/clip-vit-base-patch16/openai_imagenet_template/imagenet2012/Lipsum-FT/0/*.log
```

## License

```
MIT License

Copyright (c) 2024 cs-giung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
