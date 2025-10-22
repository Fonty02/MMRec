# coding: utf-8
# @email  : enoche.chow@gmail.com

import os
import numpy as np
import torch
import torch.nn as nn
from logging import getLogger


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError
    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # load encoded features here
        self.logger = getLogger()
        self.v_feat, self.t_feat, self.a_feat = None, None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            
            self.logger.info("\n" + "="*80)
            self.logger.info("CARICAMENTO FEATURE MULTIMODALI")
            self.logger.info("="*80)
            
            # Vision features
            if config['vision_feature_file'] is not None:
                v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
                if os.path.isfile(v_feat_file_path):
                    self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                        self.device)
                    self.logger.info(f"✓ Vision features caricate: {config['vision_feature_file']} - Shape: {self.v_feat.shape}")
                else:
                    self.logger.warning(f"✗ Vision features NON trovate: {v_feat_file_path}")
            else:
                self.logger.info("○ Vision features: NON richieste (None)")
            
            # Text features
            if config['text_feature_file'] is not None:
                t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
                if os.path.isfile(t_feat_file_path):
                    self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                        self.device)
                    self.logger.info(f"✓ Text features caricate: {config['text_feature_file']} - Shape: {self.t_feat.shape}")
                else:
                    self.logger.warning(f"✗ Text features NON trovate: {t_feat_file_path}")
            else:
                self.logger.info("○ Text features: NON richieste (None)")
            
            # Audio features
            if config['audio_feature_file'] is not None:
                a_feat_file_path = os.path.join(dataset_path, config['audio_feature_file'])
                if os.path.isfile(a_feat_file_path):
                    self.a_feat = torch.from_numpy(np.load(a_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                        self.device)
                    self.logger.info(f"✓ Audio features caricate: {config['audio_feature_file']} - Shape: {self.a_feat.shape}")
                else:
                    self.logger.warning(f"✗ Audio features NON trovate: {a_feat_file_path}")
            else:
                self.logger.info("○ Audio features: NON richieste (None)")
            
            # Riepilogo
            loaded_features = []
            if self.v_feat is not None:
                loaded_features.append("Vision")
            if self.t_feat is not None:
                loaded_features.append("Text")
            if self.a_feat is not None:
                loaded_features.append("Audio")
            
            if loaded_features:
                self.logger.info(f"\n✓ Feature caricate con successo: {', '.join(loaded_features)}")
            else:
                self.logger.error("✗ ERRORE: Nessuna feature caricata!")
            
            self.logger.info("="*80 + "\n")

            assert self.v_feat is not None or self.t_feat is not None or self.a_feat is not None, 'Features all NONE'
