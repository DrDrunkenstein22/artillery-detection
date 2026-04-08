#!/usr/bin/env bash
# scripts/setup.sh — One-shot environment setup

set -e

echo "=== Artillery Detection — Environment Setup ==="

# Create virtual env
python3 -m venv .venv   
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing Modal..."
pip install modal

echo ""
echo "=== Next Steps ==="
echo "1. Activate the environment:  source .venv/bin/activate"
echo "2. Set credentials:"
echo "   export KAGGLE_USERNAME=your_username"
echo "   export KAGGLE_KEY=your_api_key"
echo "   export ROBOFLOW_API_KEY=your_key"
echo ""
echo "3. Authenticate Modal:        modal setup"
echo ""
echo "4. (Optional) Create Modal secrets for W&B:"
echo "   modal secret create wandb-secret WANDB_API_KEY=your_key"
echo ""
echo "5. Download datasets:         python src/data/download_datasets.py"
echo "6. Merge & balance:           python src/data/merge_datasets.py"
echo "7. Audit class balance:       python src/data/class_audit.py"
echo ""
echo "8. Train all models:"
echo "   modal run modal_jobs/modal_train_yolov11.py"
echo "   modal run modal_jobs/modal_train_rtdetr.py"
echo "   modal run modal_jobs/modal_train_faster_rcnn.py"
echo ""
echo "9. Evaluate & compare:"
echo "   modal run modal_jobs/modal_evaluate.py"
echo "   python src/evaluation/compare_models.py"
echo ""
echo "Done! Check results/ for outputs."
