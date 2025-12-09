# CKPT="./checkpoints/resnet121-epoch=00-val_auroc=1.000.ckpt"
CKPT="./checkpoints/densenet121-epoch=04-val_auroc=1.000.ckpt"

BASE_CMD="python main.py test \
  --data_root chest_xray \
  --labels_csv labels.csv \
  --model_ckpt \"$CKPT\" \
  --batch_size 32 \
  --img_size 224 \
  --accelerator mps --devices 1"

combos=(
  "0 0"
  "2 0"
  "4 0"
  "6 0"
  "8 0"
  "10 0"

  "0 2"
  "0 4"
  "0 6"
  "0 8"
  "0 10"

  "2 2"
  "4 4"
  "6 6"
  "8 8"
  "10 10"
)

for combo in "${combos[@]}"; do
  gauss=$(echo "$combo" | awk '{print $1}')
  poiss=$(echo "$combo" | awk '{print $2}')

  echo
  echo "=== Running gauss=$gauss, poiss=$poiss ==="
  eval $BASE_CMD" --gauss $gauss --poiss $poiss"
done
