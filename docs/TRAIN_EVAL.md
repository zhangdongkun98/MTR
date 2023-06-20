## Training 
For example, train with 8 GPUs: 
```
cd tools

bash scripts/torchrun_train.sh 8
```
Actually, during the training process, the evaluation results will be logged to the log file under `output/waymo/mtr+100_percent_data/my_first_exp/log_train_xxxx.txt`

## Testing
For example, test with 8 GPUs: 
```
cd tools

python test.py --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --ckpt ../results/waymo/mtr+100_percent_data/2023-06-19-22\:09\:54----look/ckpt/checkpoint_epoch_20.pth --extra_tag look --batch_size 20

torchrun --nproc_per_node=8 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --ckpt ../results/waymo/mtr+100_percent_data/2023-06-19-22\:09\:54----look/ckpt/checkpoint_epoch_20.pth --extra_tag look --batch_size 20
```


## Vis

```
python vis_dataset.py --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 20
```
