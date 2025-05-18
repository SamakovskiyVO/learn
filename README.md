# Frame-Quality Training Suite

Этот репозиторий содержит **один ноутбук** `all_experiments.ipynb`, где
последовательно реализованы **8 улучшений** MobileNetV3-Small,
описанных в презентации диплома:

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
git clone <repo-url> frame-quality-training
cd frame-quality-training

# создать виртуальное окружение
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (если у вас только .py) → .ipynb
pip install jupytext
jupytext --to notebook all_experiments.py

jupyter lab
