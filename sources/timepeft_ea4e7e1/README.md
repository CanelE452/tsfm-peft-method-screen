# Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models

[![Python 3.10](https://img.shields.io/badge/python-3.10.15-blue.svg)](https://www.python.org/downloads/release/python-31015/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.13.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)


## 📖 Overview

Recent studies have attempted to fine-tune time-series foundation models to enhance a target dataset's forecasting performance. However, these approaches proceed without a clear criterion for identifying complex datasets that require fine-tuning due to performance degradation in zero-shot forecasting. To distinguish datasets that are more challenging than standard benchmarks, we introduce data-driven temporal complexity and multichannel complexity. Temporal complexity captures the difficulty of identifying distinct patterns by quantifying spectral entropy in the frequency domain, while multichannel complexity captures cross-channel information flow that can impact predictive uncertainty. These metrics serve as effective proxies for performance gains achievable through fine-tuning. Based on the two metrics, we develop Time-PEFT, a parameter-efficient fine-tuning framework that incorporates a frequency adapter for top- filtering and a channel adapter for multichannel modeling. With the base variant of MOMENT as a backbone, Time-PEFT improves performance by up to 38% over LoRA on complex datasets.


## 🔨 Environment Setup

**Install Dependencies:**
Please refer to `requirements.txt`

```bash
    pip install torch==1.13.1+cu117 --extra-index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)
    pip install -r requirements.txt
```

## 🚀 How to Run

```bash
cd /path/to/TimePEFT
python run.py \
  --data-root /datasets/ETT \
  --data-path ETTh1.csv \
  --dataset-name ETTh1 \
  --device cuda \
  --modelsize small \
  --seq-len 96 --pred-len 96
```

## 📊 Results

After training, the evaluation metrics (MSE, MAE) and model logs will be saved  as a JSON file.
Example Output:
```json
{
  "val_metrics": {
    "mse": 0.000,
    "mae": 0.000
  },
  "test_metrics": {
    "mse": 0.000,
    "mae": 0.000
  }
}
```

## ⚙️ Citation

If you use this research, please cite our paper:

```
@inproceedings{na2026timepeft,
  title={Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models},
  author={Jihye Na, Patara Trirat, Chanyoung Park and Jae-Gil Lee},
  booktitle={ICML},
  year={2026}
}
```


## 🔗 Code Reference & Acknowledgements

We appreciate the following open-source works:

* [TSLib](https://github.com/thuml/Time-Series-Library)
* [MOMENT](https://github.com/moment-timeseries-foundation-model/moment)
