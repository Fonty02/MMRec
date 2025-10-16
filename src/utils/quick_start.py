# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import csv
from datetime import datetime


def save_results_to_csv(config, hyper_ret, best_test_idx, csv_filename):
    """
    Salva i risultati dell'addestramento in un file CSV
    """
    # Crea la directory reports se non esiste
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'reports')
    if not os.path.exists(reports_dir):
        reports_dir = os.path.join(os.getcwd(), 'reports')
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
    
    csv_path = os.path.join(reports_dir, csv_filename)
    
    # Prepara le intestazioni
    fieldnames = ['timestamp', 'model', 'dataset', 'run_id', 'is_best']
    
    # Aggiungi i nomi degli iperparametri
    for param in config['hyper_parameters']:
        fieldnames.append(param)
    
    # Aggiungi le metriche di validazione e test
    if hyper_ret:
        valid_metrics = list(hyper_ret[0][1].keys())
        test_metrics = list(hyper_ret[0][2].keys())
        
        for metric in valid_metrics:
            fieldnames.append(f'valid_{metric}')
        for metric in test_metrics:
            fieldnames.append(f'test_{metric}')
    
    # Scrivi i risultati nel CSV
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Scrivi l'header solo se il file è nuovo
        if not file_exists:
            writer.writeheader()
        
        # Scrivi ogni risultato
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for idx, (hyper_tuple, valid_result, test_result) in enumerate(hyper_ret):
            row = {
                'timestamp': timestamp,
                'model': config['model'],
                'dataset': config['dataset'],
                'run_id': idx + 1,
                'is_best': 'Yes' if idx == best_test_idx else 'No'
            }
            
            # Aggiungi gli iperparametri
            for param, value in zip(config['hyper_parameters'], hyper_tuple):
                row[param] = value
            
            # Aggiungi le metriche di validazione
            for metric, value in valid_result.items():
                row[f'valid_{metric}'] = value
            
            # Aggiungi le metriche di test
            for metric, value in test_result.items():
                row[f'test_{metric}'] = value
            
            writer.writerow(row)
    
    return csv_path


def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

    # Salva i risultati in CSV
    csv_filename = f"training_results_{config['model']}_{config['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = save_results_to_csv(config, hyper_ret, best_test_idx, csv_filename)
    logger.info(f'\n📊 Results saved to CSV: {csv_path}')


