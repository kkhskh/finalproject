# CS242 Final Project - Colab Training Script
# 
# INSTRUCTIONS:
# 1. Go to https://colab.research.google.com/
# 2. Create a new notebook
# 3. Go to Runtime → Change runtime type → GPU (T4 or better)
# 4. Copy each section below into separate code cells and run them

# ====================
# CELL 1: Check GPU
# ====================
!nvidia-smi

# ====================
# CELL 2: Clone Repo
# ====================
!git clone https://github.com/kkhskh/finalproject.git
%cd finalproject

# ====================
# CELL 3: Install Dependencies
# ====================
!pip install -q -r requirements.txt

# ====================
# CELL 4: Train with Best Config (Evolved Architecture)
# ====================
!python 1.py --mode train_best --epochs 10 --config_json best_config.json

# ====================
# CELL 5: Train Manual Baseline
# ====================
!python 1.py --mode train_manual --epochs 10

# ====================
# CELL 6: Download Results (Optional)
# ====================
from google.colab import files
import os

if os.path.exists('evolved_best_model.pt'):
    files.download('evolved_best_model.pt')
    
if os.path.exists('baseline_model.pt'):
    files.download('baseline_model.pt')

