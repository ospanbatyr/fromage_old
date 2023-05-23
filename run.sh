# just to test the new dataloader, change max_len and num_workers in training

CUDA_LAUNCH_BLOCKING=1 python -u main.py --dist-backend 'nccl' --world-size 1 --rank 0 --dataset=CheXpert  --val-dataset=CheXpert --opt-version='facebook/opt-350m' --visual-model='openai/clip-vit-large-patch14' --exp_name='fromage_exp' --image-dir='data/'  --log-base-dir='runs/' --batch-size=64  --val-batch-size=32  --learning-rate=0.0003 --precision='fp32' --print-freq=100 --workers=2 --image-dir='/userfiles/oince22/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/CheXpert-v1.0' --max-len=36
