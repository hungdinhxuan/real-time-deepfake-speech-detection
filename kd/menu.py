import argparse


def get_main_menu():
    parser = argparse.ArgumentParser(
        description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/datad/hungdx/KDW2V-AASISTL/databases/',
                        help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- DF
    %      |- ASVspoof2021_DF_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='/datab/hungdx/KDW2V-AASISTL/protocols/',
                        help='Change with path to user\'s DF database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
 
    %   |- ASVspoof_DF_cm_protocols
    %      |- ASVspoof2021.DF.cm.eval.trl.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--yaml', type=str, default='config.yaml',
                        help='YAML file for hyperparameters')

    parser.add_argument('--dataset', type=str, default='DF21',
                        help='Dataset for evaluation')
    parser.add_argument('--batch_size', type=int, default=14)
    # can change the defult like5
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--padding', type=float, default=4.0375)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--patience', type=int, default=7,
                        help='Patience for early stopping')
    # model
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--padding_size', type=int, default=64600,
                        help='padding size (default: 64600)')
    parser.add_argument('--workers', type=int, default=8,
                        help='numbers of worker')

    parser.add_argument('--model_path', type=str,
                        default='./W2V2-AASIST-teacher.pth', help='Model checkpoint')
    parser.add_argument('--student_model_path', type=str,
                        default=None, help='Student model checkpoint')
    parser.add_argument('--student_model_type', type=str,
                        default='SelfDistil_W2V2BASE_AASISTL', help='Type of student model')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--student_ckpt', type=str,
                        help='Student checkpoint', default='')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='DF',
                        choices=['LA', 'PA', 'DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--wrapper_ssl', action='store_true', default=False,
                        help='Wrapper ssl model to torchaudio')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='use amp mode')
    parser.add_argument('--qat', action='store_true', default=False,
                        help='Enable Quantization Aware Training')
    parser.add_argument('--bf16', action='store_true', default=False,
                        help='Enable BF16 inference')
    parser.add_argument('--is_eval', action='store_true',
                        default=False, help='eval database')
    parser.add_argument('--is_eval_teacher', action='store_true',
                        default=False, help='eval teacher')
    parser.add_argument('--scale_export', action='store_true',
                        default=False, help='Export scaled version')

    parser.add_argument('--batch_size_eval', type=int, default=14)
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false',
                        default=True,
                        help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true',
                        default=False,
                        help='use cudnn-benchmark? (default false)')
    parser.add_argument('--num_eval_samples', type=int,
                        default=150000, help='Number of evaluation samples')
    parser.add_argument('--student_restore', action='store_true', default=False,
                        help='Student model checkpoint')

    parser.add_argument('--ssl_type', type=str,
                        help='Type of SSL models', default='Distil_XLSR')

    parser.add_argument('--half',  help='eval with half precision',
                        action='store_true', default=False)

    parser.add_argument('--onnx',  help='Export onnx',
                        action='store_true', default=False)

    parser.add_argument('--self_KD_type', type=str,
                        help='Type of Self KD models', default='selfKD')

    parser.add_argument('--self_KD', action='store_true', default=False,
                        help='Self KD')

    parser.add_argument('--KD_logits', action='store_true', default=False,
                        help='KD_logits')

    parser.add_argument('--KD_cosine', action='store_true', default=False,
                        help='KD cosine loss')

    parser.add_argument('--KD_mse', action='store_true', default=False,
                        help='KD mse loss')

    ## ===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=3,
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5,
                        help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20,
                        help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000,
                        help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100,
                        help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000,
                        help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10,
                        help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100,
                        help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0,
                        help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0,
                        help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                        help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                        help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5,
                        help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10,
                        help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2,
                        help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                        help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40,
                        help='Maximum SNR value for coloured additive noise.[defaul=40]')
    ## ===================================================Rawboost data augmentation ======================================================================#
    return parser.parse_args()


def get_main_benchmark_cpu():
    parser = argparse.ArgumentParser(description='Benchmark inference on CPU')
    parser.add_argument('--database_path', type=str, default='./databases/',
                        help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.')
    parser.add_argument('--protocols_path', type=str, default='./protocols/',
                        help='Change with path to user\'s DF database protocols directory address')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--model_path', type=str,
                        default='./W2V2-AASIST-teacher.pth', help='Model checkpoint')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_eval_samples', type=int,
                        default=1000, help='Number of evaluation samples')
    parser.add_argument('--num_threads', type=int,
                        default=1, help='Number of threads')

    parser.add_argument('--num_warmup', type=int, default=5,
                        help='Number of warmup runs')

    parser.add_argument('--is_quant', action='store_true', default=False,
                        help='Quantization inference mode')

    parser.add_argument('--KD_logits', action='store_true', default=False,
                        help='Distillation loss is calculated from the logits of the networks')

    parser.add_argument('--KD_cosine', action='store_true', default=False,
                        help='Distillation loss is calculated from Cosine loss minimization')

    parser.add_argument('--KD_mse', action='store_true', default=False,

                        help='Distillation loss is calculated from Cosine loss minimization')
    return parser.parse_args()


def get_hyp_tuning_menu():
    parser = argparse.ArgumentParser(
        description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/datab/hungdx/KDW2V-AASISTL/databases/',
                        help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- DF
    %      |- ASVspoof2021_DF_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='/datab/hungdx/KDW2V-AASISTL/protocols/',
                        help='Change with path to user\'s DF database protocols directory address')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')

    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')

    parser.add_argument('--model_path', type=str,
                        default='/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth', help='Model checkpoint')

    parser.add_argument('--cudnn-deterministic-toggle', action='store_false',
                        default=True,
                        help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true',
                        default=False,
                        help='use cudnn-benchmark? (default false)')

    parser.add_argument('--student_restore', action='store_true', default=False,
                        help='Student model checkpoint')

    parser.add_argument('--KD_logits', action='store_true', default=False,
                        help='Distillation loss is calculated from the logits of the networks')

    parser.add_argument('--KD_cosine', action='store_true', default=False,
                        help='Distillation loss is calculated from Cosine loss minimization')

    parser.add_argument('--KD_mse', action='store_true', default=False,
                        help='Distillation loss is calculated from Cosine loss minimization')
    ## ===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=3,
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5,
                        help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20,
                        help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000,
                        help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100,
                        help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000,
                        help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10,
                        help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100,
                        help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0,
                        help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0,
                        help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                        help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                        help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5,
                        help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10,
                        help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2,
                        help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                        help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40,
                        help='Maximum SNR value for coloured additive noise.[defaul=40]')
    ## ===================================================Rawboost data augmentation ======================================================================#

    return parser.parse_args()
