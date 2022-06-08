# Model
---
- Local Damage Detector (LDD)
- Semi-Supervised Learning (SSL)
---
### Used Model

- Unet
- EfficientUnetb0
- EfficientUnetb4
- EfficientUnetb7
---
### Install



### How to Run the Code 


SSL train
```
!python 'file path'/train.py \
--lr 'float' \
--batch_size 'int' \
--num_epoch 'int'  \
--optimizer "str" \
--image_size 'int' \
--weight_decay 'float' \
--wandb_project_name "str" \
--wandb_entity_name "str" \
--wandb_name "str" \
--base_path "data_path" \
--model_save_name "model save path" \
--class_name 'str' \
--model 'model name' \
--loss_fn 'str' \
--focal_alpha 'float' \
--focal_gamma 'float' \
--both_weight_a 'float' \
--both_weight_b 'float'
```

SSL test
```
!python 'file path'/test.py \
--image_size 'int' \
--wandb_project_name "str" \
--wandb_entity_name "str" \
--wandb_name "str" \
--base_path 'data path' \
--model_load_path "model.pth path" \
--class_name 'str' \
--model 'model name'
```

Eval (make pseudo mask)
```
!python 'file path'/eval.py \
--image_size 'int' \
--wandb_project_name "str" \
--wandb_entity_name "str" \
--wandb_name "str" \
--base_path 'data path' \
--model_load_path "model.pth path" \
--save_result_dir "save mask path" \
--class_name 'str' \
--model 'model name'
```

LDD train 
```
!python 'file path'/train.py \
--lr 'int' \
--batch_size 'int' \
--num_epoch 'int' \
--optimizer 'str' \
--image_size 'int' \
--wandb_project_name "str" \
--wandb_entity_name "str" \
--wandb_name "str" \
--base_path "data path" \
--model_save_name "model save path" \
--model 'model name'
```

LDD test
```
!python 'file path'/test.py \
--image_size 'int' \
--wandb_project_name "str" \
--wandb_entity_name "str" \
--wandb_name "str" \
--base_path 'data path' \
--model_load_path "model.pth path" \
--model 'model name'
```