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
import gc
import torch


def cleanup_gpu_memory():
    """
    Cleans GPU and CPU memory between hyperparameter runs
    Prevents memory leaks and segmentation faults
    """
    try:
        gc.collect()  # Python garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Empty CUDA cache
            torch.cuda.synchronize()  # Synchronize with GPU
    except Exception as e:
        print(f"[WARNING] Error during GPU cleanup: {e}")


def save_results_to_csv(config, best_valid_result, best_test_result, csv_filename):
    """
    Saves ONLY the results of the best model to a CSV file
    """
    print(f"\n[DEBUG] save_results_to_csv called:")
    print(f"  - best_valid_result: {best_valid_result}")
    print(f"  - best_test_result: {best_test_result}")
    
    # Create the reports directory if it doesn't exist
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'reports')
    if not os.path.exists(reports_dir):
        reports_dir = os.path.join(os.getcwd(), 'reports')
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
    
    csv_path = os.path.join(reports_dir, csv_filename)
    
    # Prepare the headers
    # Always include the experiment name and feature columns so that
    # the CSV header is consistent even on the first write.
    fieldnames = ['timestamp', 'experiment_name', 'model', 'dataset',
                  'vision_feature', 'text_feature', 'audio_feature']
    
    # Add hyperparameter names
    for param in config['hyper_parameters']:
        fieldnames.append(param)
    
    # Add validation and test metrics
    valid_metrics = list(best_valid_result.keys())
    test_metrics = list(best_test_result.keys())
    
    for metric in valid_metrics:
        fieldnames.append(f'valid_{metric}')
    for metric in test_metrics:
        fieldnames.append(f'test_{metric}')
    
    # Write the results to CSV
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        # Prepare the row
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = {
            'timestamp': timestamp,
            'model': config['model'],
            'dataset': config['dataset']
        }
        
        # Add feature configuration (always present as columns)
        row['experiment_name'] = config['experiment_name'] if config['experiment_name'] is not None else 'None'
        row['vision_feature'] = config['vision_feature_file'] if config['vision_feature_file'] is not None else 'None'
        row['text_feature'] = config['text_feature_file'] if config['text_feature_file'] is not None else 'None'
        row['audio_feature'] = config['audio_feature_file'] if config['audio_feature_file'] is not None else 'None'
        
        # Add the best hyperparameters
        # Hyperparameters are already set in config during the loop
        for param in config['hyper_parameters']:
            row[param] = config[param]
        
        # Add validation metrics
        for metric, value in best_valid_result.items():
            row[f'valid_{metric}'] = value
        
        # Add test metrics
        for metric, value in best_test_result.items():
            row[f'test_{metric}'] = value
        
        writer.writerow(row)
        csvfile.flush()  # Force write to disk
    
    # Confirmation log
    print(f"✓ Results written to: {csv_path}")
    
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
    

        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        
        # Clean GPU and CPU memory
        cleanup_gpu_memory()
        logger.debug(f'[Cleanup] Resources freed before next run')

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

    # ===== FINAL CLEANUP =====
    cleanup_gpu_memory()
    logger.info('[Cleanup] Final cleanup completed')

    # Save ONLY the results of the best model to CSV
    csv_filename = f"best_results_{config['model']}_{config['dataset']}.csv"
    best_valid_result = hyper_ret[best_test_idx][1]
    best_test_result = hyper_ret[best_test_idx][2]
    csv_path = save_results_to_csv(config, best_valid_result, best_test_result, csv_filename)
    logger.info(f'\n📊 Best results saved to CSV: {csv_path}')


