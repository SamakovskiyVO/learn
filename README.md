# Frame-Quality Training Suite

Этот репозиторий содержит ноутбук `all_experiments.ipynb`, где
последовательно реализованы эксперемены по обучению MobileNetV3-Small,
для проекта AutoVisionTune:

1. Baseline + SGD (macro-F1 ≈ 0.77)  
2. AdamW + ReduceLROnPlateau  
3. Class Weighting  
4. Warm-up LR + CosineAnnealingRestarts  
5. Feature-level Mixup  
6. DropBlock вместо Dropout  
7. Label Smoothing 0.05  
8. SWAG (усреднение весов) → macro-F1 ≈ 0.915  

## Быстрый старт

```bash

# создать виртуальное окружение
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pip install jupytext
jupytext --to notebook all_experiments.py

jupyter lab
