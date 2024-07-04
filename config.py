class SysConfig:

    def __init__(self, config):
        self.load_config(config)

    def load_config(self, config):

        self.wandb_disabled = config.get('wandb_disabled', False)
        self.wandb_project = config.get('wandb_project', 'ASV-Spoofing')
        self.wandb_name = config.get('wandb_name', 'SE-Rawformer, no DA')
        self.wandb_entity = config.get('wandb_entity', 'hungdinhxdev')
        self.wandb_key = config.get('wandb_key', '')
        self.wandb_notes = config.get('wandb_notes', '')

        self.path_label_asv_spoof_2019_la_train = config.get(
            'path_label_asv_spoof_2019_la_train', '')
        self.path_label_asv_spoof_2019_la_dev = config.get(
            'path_label_asv_spoof_2019_la_dev', '')
        self.path_asv_spoof_2019_la_train = config.get(
            'path_asv_spoof_2019_la_train', '')
        self.path_asv_spoof_2019_la_dev = config.get(
            'path_asv_spoof_2019_la_dev', '')
        self.path_label_asv_spoof_2019_la_eval = config.get(
            'path_label_asv_spoof_2019_la_eval', '')
        self.path_asv_spoof_2019_la_eval = config.get(
            'path_asv_spoof_2019_la_eval', '')

        self.path_label_asv_spoof_2021_la_eval = config.get(
            'path_label_asv_spoof_2021_la_eval', '')
        self.path_label_asv_spoof_2021_la_eval_spec = config.get(
            'path_label_asv_spoof_2021_la_eval_spec', False)
        self.path_asv_spoof_2021_la_eval = config.get(
            'path_asv_spoof_2021_la_eval', '')

        self.path_asv_spoof_2021_df_eval = config.get(
            'path_asv_spoof_2021_df_eval', '')
        self.path_label_asv_spoof_2021_df_eval = config.get(
            'path_label_asv_spoof_2021_df_eval', '')

        self.num_workers = config.get('num_workers', 4)

        self.path_to_save_model = config.get('path_to_save_model', './runs')
        self.df21_score_save_path = config.get(
            'df21_score_save_path', './runs')
        self.la21_score_save_path = config.get(
            'la21_score_save_path', './runs')
        self.la19_score_save_path = config.get(
            'la19_score_save_path', './runs')

        self.path_itw_eval = config.get('path_itw_eval', '')
        self.path_label_itw_eval = config.get(
            'path_label_itw_eval', '')

        self.model = config.get('model', 'XLSR_AASIST')
        self.student_model = config.get('student_model', 'XLSR_AASIST')

    def __str__(self):
        return f"SysConfig: {self.__dict__}"

class ExpConfig:

    def __init__(self, config):
        self.load_config(config)

    def load_config(self, config):

        self.random_seed = config.get('random_seed', 1024)
        self.is_pre_emphasis = config.get('is_pre_emphasis', True)
        self.is_random_start = config.get('is_random_start', False)
        self.include_non_speech = config.get('include_non_speech', True)
        self.include_residual = config.get('include_residual', True)
        self.pre_emphasis = config.get('pre_emphasis', 0.97)
        self.sample_rate = config.get('sample_rate', 16000)
        self.train_duration_sec = config.get('train_duration_sec', 4)
        self.test_duration_sec = config.get('test_duration_sec', 4)
        self.batch_size_train = config.get('batch_size_train', 32)
        self.batch_size_test = config.get('batch_size_test', 40)
        self.lr = config.get('lr', 0.000001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.max_epoch = config.get('max_epoch', 100)
        self.allow_data_augmentation = config.get(
            'allow_data_augmentation', False)
        self.data_augmentation = config.get('data_augmentation', ['ACN'])
        self.restore_checkpoint = config.get('restore_checkpoint', None)
        self.kwargs = config.get('kwargs', {})
        self.kd_kwargs = config.get('kd_kwargs', {})

    def __str__(self):
        return f"ExpConfig: {self.__dict__}"