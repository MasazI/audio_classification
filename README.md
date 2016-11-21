# audio_classification
audio classification is an implementation MLP for audio classification using TensorFlow

## requirements
- TensorFlow 0.10+
- Numpy
- Matplotlib
- Librosa

## how to use
- you can use 'train.pkl' and 'test.pkl'. instead, download UrbanSound8k dataset.

`python task.py`

## save model
`output/ckpt-N`

## tensorboard
<img src=https://github.com/MasazI/audio_classification/blob/master/output/debug/tb.png/>

## audio wave image
<img src="https://github.com/MasazI/audio_classification/blob/master/output/debug/f1_waveplot.jpg" width="300"/>
<img src="https://github.com/MasazI/audio_classification/blob/master/output/debug/f2_spec.jpg" width="300"/>
<img src="https://github.com/MasazI/audio_classification/blob/master/output/debug/f3_logpowerspec.jpg" width="300"/>

---

Copyright (c) 2015 Masahiro Imai
Released under the MIT license
