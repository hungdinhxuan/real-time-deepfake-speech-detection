#!/bin/bash
# Install dependency for fairseq

# Name of the conda environment
ENVNAME=realtime_add

eval "$(conda shell.bash hook)"
conda activate ${ENVNAME}
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Install conda environment ${ENVNAME}"
    
    # conda env
    conda create -n ${ENVNAME} python=3.9 pip --yes
    conda activate ${ENVNAME}
    echo "===========Install pytorch==========="
    pip install torch==2.3.0 torchaudio==2.3.0 
    # install librosa
    pip install librosa==0.10.2

    # install tensorboard
    pip install tensorboardx==2.6.2.2

    # install librosa
    pip install librosa==0.10.0

    # install tqdm
    pip install tqdm==4.66.4
    
    # install pandas
    pip install pandas==2.0.1

    # install matplotlib
    pip install matplotlib==3.7.0

    # install wandb
    pip install wandb==0.17.0

    


else
    echo "Conda environment ${ENVNAME} has been installed"
fi
