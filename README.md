## Training

## Run

```
PYTHONPATH=$PYTHONPATH:/datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python main.py --config "confs/test_tmp/1s_aasist_r4_w_o_silence_re_init_3layers.yml"
```


PYTHONPATH=$PYTHONPATH:/datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=1 python main.py --config "confs/test_tmp/1s_aasist_r4_w_o_silence.yml" --is_eval --ckpt='aasist_runs/XLSR_AASIST_RawBoost4_freeze_last12/best_LA_epoch77_0.005686_99.5612.pt' --is_score --tracks DF21 --comment="best77"

## EER
```
python main.py --cm-score-file /datad/hungdx/Rawformer-implementation-anti-spoofing/XLSRraw4_best32_and_Conformer_best50_1s_fusion.txt --track DF --subset eval
```

## Evaluation
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=3 python main.py --config confs/test_tmp/1s_aasist_r4_w_o_silence_re_init_3layers.yml --is_eval --is_score --tracks DF21 --score_all_folder_path=aasist_runs/XLSR_AASIST_RawBoost4_freeze_last3
```


## Evaluation using distilled model
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=3 python main_kd.py --config kd_configs/test/1s_my_kd_aasist_df21_1s.yml --is_eval --is_score --tracks DF21 --score_all_folder_path best_pretrained/iconip2024_runs_newmap_g2/Distil_XLSR_5_Custom_Trans_Layer_AASIST_from_bestr4_CosineAnnealing_MSE_aug4_newmap_g_2 --comment='_Distil_XLSR_5_Custom_Trans_Layer_AASIST_from_bestr4_CosineAnnealing_MSE_aug4_newmap_g_2' --eval student
```

## Evaluation folder using distilled model
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=3 python main_kd.py --config kd_configs/test/1s_my_kd_aasist_df21_1s.yml --is_eval --is_score --tracks DF21 --ckpt /data/best_pretrained/Distil_XLSR_5_Custom_Trans_Layer_AASIST_from_bestr4_CosineAnnealing_MSE_aug4_newmap_g_2_best32.pth --comment='_Distil_XLSR_5_Custom_Trans_Layer_AASIST_from_bestr4_CosineAnnealing_MSE_aug4_newmap_g_2_best32' --eval student
```

### Training with knowledge distillation
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main_kd.py --config kd_configs/1s_aasist_aug4_w_o_silence.yml --ckpt /datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/runs/aasist_runs/XLSR_AASIST_RawBoost4/best_LA_epoch32_0.004826_99.6229.pt
```

