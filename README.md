## Training

## Run

```
PYTHONPATH=$PYTHONPATH:/datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python main.py --config "confs/1s_conformer_r4_w_o_silence.yml"
```

## Accuracy
```
PYTHONPATH=$PYTHONPATH:/datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=3 python main.py --config "confs/4s_conformer_baseline.yml" --accuracy --ckpt='pretrained/conformer_best.pth'
```


## Evaluation on ASVSpoof5
```
PYTHONPATH=$PYTHONPATH:/datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=3 python main.py --config "confs/4s_conformer_baseline.yml" --is_eval --is_score --ckpt='pretrained/conformer_best.pth' --track ASVSpoof5
```

```
PYTHONPATH=$PYTHONPATH:/datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python main.py --config "confs/4s_aasist_baseline.yml" --is_eval --is_score --ckpt='pretrained/Best_LA_model_for_DF.pth' --track ASVSpoof5
```


PYTHONPATH=$PYTHONPATH:/datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=1 python main.py --config "confs/test_tmp/1s_aasist_r4_w_o_silence.yml" --is_eval --ckpt='aasist_runs/XLSR_AASIST_RawBoost4_freeze_last12/best_LA_epoch77_0.005686_99.5612.pt' --is_score --tracks DF21 --comment="best77"

## EER
```
python main.py --cm-score-file /data/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/KD_ICONIP/XLSR-6-AASIST_LA19-1s_conf-1-ce/best_checkpoint_148.pth_DF21.txt --track DF --subset eval
```
****
## Evaluation
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=3 python main.py --config confs/1s_conformer_r4_w_o_silence.yml --is_eval --is_score --tracks DF21 --score_all_folder_path=runs/aasist_runs/ConformerModel_RawBoost4
```


## Evaluation folder using distilled model
```
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 python main_kd.py --config kd_configs/test/1s_my_kd_aasist_df21_1s_6l.yml --is_eval --is_score --tracks DF21 --score_all_folder_path KD_ICONIP/XLSR-6-AASIST_LA19-1s_conf-1-5-c --eval student
```

## Evaluation  using distilled model
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main_kd.py --config kd_configs/test/1s_my_kd_conformer_df21new_1s_6l.yml --is_eval --is_score --tracks DF21,InTheWild,InTheWild1s --ckpt KD_ICONIP/XLSR-6-Conformer_LA19-1s_conf-1-5_r4teacher/best_checkpoint_125.pth --comment='_XLSR-6-Conformer_LA19-1s_conf-1-5_r4teacher_best125_new1s' --eval student
```


## Evaluation KD 
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main_kd.py --config kd_configs/test/1s_my_kd_aasist_df21_1s_6l.yml --is_eval --is_score --tracks InTheWild,InTheWild1s --ckpt pretrained/Best_LA_model_for_DF.pth --comment='_aasist_baseline_best' --eval teacher
```

### Training with knowledge distillation
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python main_kd.py --config kd_configs/1s_aasist_aug4_w_o_silence.yml --ckpt /datad/hungdx/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/runs/aasist_runs/XLSR_AASIST_RawBoost4/best_LA_epoch32_0.004826_99.6229.pt
```

