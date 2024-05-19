PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq OMP_NUM_THREADS=5 CUDA_VISIBLE_DEVICES=2 python main.py --config "confs/1s_aasist_r5_w_o_silence.yml" --is_eval --score_all_folder_path='./runs/aasist_runs/XLSR_AASIST_RawBoost5' --is_score --tracks LA19,LA21 

## EER
```
python main.py --cm-score-file /home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/runs/aasist_runs/XLSR_AASIST_RawBoost4/scores/random/df21/XLSR_AASIST_DA3_EMPHASIS_DF21_score_epoch32_0.004826_99.6229.pt.txt --track DF --subset eval
```

## Evaluation
```
PYTHONPATH=$PYTHONPATH:/home/hungdx/code/Towards-Real-Time-Deepfake-Speech-Detection-in-Resource-Limited-Scenarios/fairseq CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python main.py --config confs/test_on_new_set/1s_aasist_r4_w_o_silence.yml --is_eval --is_score --tracks DF21,InTheWild --ckpt runs/aasist_runs/XLSR_AASIST_RawBoost4/best_LA_epoch32_0.004826_99.6229.pt --comment '_best32'
```