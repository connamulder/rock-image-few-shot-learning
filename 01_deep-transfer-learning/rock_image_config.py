"""
    @Project: rock-image-transfer-learning
    @File   : rock_image_config.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2022-10-14
    @info   : Define a class for program parameter configuration.
              The class responsible for reading program parameter configurations.
"""


import sys


class RockImageConfigure:
    def __init__(self, config_file='system.config'):
        config = self.config_file_to_dict(config_file)

        # Model Configuration:
        the_item = 'image_size'
        if the_item in config:
            self.image_size = int(config[the_item])
        the_item = 'image_class'
        if the_item in config:
            self.image_class = int(config[the_item])
        the_item = 'is_similarity_metric'
        if the_item in config:
            self.is_similarity_metric = self.str2bool(config[the_item])

        # Training Settings:
        the_item = 'epoch'
        if the_item in config:
            self.epoch = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])
        the_item = 'dropout'
        if the_item in config:
            self.dropout = float(config[the_item])
        the_item = 'learning_rate'
        if the_item in config:
            self.learning_rate = float(config[the_item])

        the_item = 'checkpoints_max_to_keep'
        if the_item in config:
            self.checkpoints_max_to_keep = int(config[the_item])
        the_item = 'print_per_batch'
        if the_item in config:
            self.print_per_batch = int(config[the_item])

        the_item = 'is_early_stop'
        if the_item in config:
            self.is_early_stop = self.str2bool(config[the_item])
        the_item = 'patient'
        if the_item in config:
            self.patient = int(config[the_item])

        the_item = 'do_train'
        if the_item in config:
            self.do_train = self.str2bool(config[the_item])
        the_item = 'do_val'
        if the_item in config:
            self.do_val = self.str2bool(config[the_item])
        the_item = 'do_test'
        if the_item in config:
            self.do_test = self.str2bool(config[the_item])
        the_item = 'is_scratch_train'
        if the_item in config:
            self.is_scratch_train = self.str2bool(config[the_item])

        # Datasets(Input/Output):
        the_item = 'data_folder'
        if the_item in config:
            self.data_folder = config[the_item]
        the_item = 'checkpoints_dir'
        if the_item in config:
            self.checkpoints_dir = config[the_item]
        the_item = 'checkpoint_name'
        if the_item in config:
            self.checkpoint_name = config[the_item]

    @staticmethod
    def config_file_to_dict(input_file):
        config = {}
        fins = open(input_file, 'r', encoding='utf-8').readlines()
        for line in fins:
            if len(line) > 0 and line[0] == '#':
                continue
            if '=' in line:
                pair = line.strip().split('#', 1)[0].split('=', 1)
                item = pair[0]
                value = pair[1]
                # noinspection PyBroadException
                try:
                    if item in config:
                        print('Warning: duplicated config item found: {}, updated.'.format((pair[0])))
                    if value[0] == '[' and value[-1] == ']':
                        value_items = list(value[1:-1].split(','))
                        config[item] = value_items
                    else:
                        config[item] = value
                except Exception:
                    print('configuration parsing error, please check correctness of the config file.')
                    exit(1)
        return config

    @staticmethod
    def str2bool(string):
        if string == 'True' or string == 'true' or string == 'TRUE':
            return True
        else:
            return False

    @staticmethod
    def str2none(string):
        if string == 'None' or string == 'none' or string == 'NONE':
            return None
        else:
            return string

    def show_data_summary(self, logger):
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY' + '++' * 20)
        logger.info(' Model Configuration:')
        logger.info('     image            size: {}'.format(self.image_size))
        logger.info('     image           class: {}'.format(self.image_class))
        logger.info('     is  similarity metric: {}'.format(self.is_similarity_metric))
        logger.info(' ' + '++' * 20)
        logger.info(' Training Settings:')
        logger.info('     epoch                : {}'.format(self.epoch))
        logger.info('     batch            size: {}'.format(self.batch_size))
        logger.info('     dropout              : {}'.format(self.dropout))
        logger.info('     learning         rate: {}'.format(self.learning_rate))
        logger.info('     max       checkpoints: {}'.format(self.checkpoints_max_to_keep))
        logger.info('     print       per_batch: {}'.format(self.print_per_batch))
        logger.info('     is     early     stop: {}'.format(self.is_early_stop))
        logger.info('     patient              : {}'.format(self.patient))
        logger.info('     is        do training: {}'.format(self.do_train))
        logger.info('     is      do validation: {}'.format(self.do_val))
        logger.info('     is         do testing: {}'.format(self.do_test))
        logger.info('     is   scratch training: {}'.format(self.is_scratch_train))
        logger.info(' ' + '++' * 20)
        logger.info(' Datasets:')
        logger.info('     data           folder: {}'.format(self.data_folder))
        logger.info('     checkpoints       dir: {}'.format(self.checkpoints_dir))
        logger.info('     checkpoint       name: {}'.format(self.checkpoint_name))
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY END' + '++' * 20)
        sys.stdout.flush()


if __name__ == '__main__':
    import logging

    configs = RockImageConfigure()
    logging.basicConfig(level=logging.INFO)
    configs.show_data_summary(logging)