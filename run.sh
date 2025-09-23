# 1. warm-up: train selection tokens
python train_rclip_v2.py --epochs 5 --save_path=OUTPUT/rclip_v2-select --fp16 --shuffle --train_select --lr=5e-4 --dataset=FashionGen
# 2. train E2 main model
python train_rclip_v2.py --epochs 20 --save_path=OUTPUT/rclip_v2 --fp16 --shuffle --lr=5e-5 --dataset=FashionGen --select_path=OUTPUT/rclip_v2-select/best.pth
