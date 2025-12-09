#coding: utf-8
"""
Script to run all configured experiments
"""

import os
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'
"""
 {
            
            'name': 'FREEDOM_text_minilm',
            'model': 'FREEDOM',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': None,   
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': None,   
            }
        },
                {
            'name': 'FREEDOM_audio_vggish',
            'model': 'FREEDOM',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': None,
                'text_feature_file': None,
                'audio_feature_file': 'audio_vggish.npy',
            }
        },

         {
            'name': 'FREEDOM_image_vit',
            'model': 'FREEDOM',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': None,
                'audio_feature_file': None,
            }
        },
        {
            'name': 'FREEDOM_text_minilm_image_vit',
            'model': 'FREEDOM',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': None,
            }
        },
        {
            'name': 'FREEDOM_text_clip_image_clip',
            'model': 'FREEDOM',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_clip.npy',
                'text_feature_file': 'text_clip.npy',
                'audio_feature_file': None,
            }
        },
        {
            
            'name': 'FREEDOM_text_minilm',
            'model': 'FREEDOM',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': None,  
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': None,  
            }
        },
        {
            'name': 'FREEDOM_image_vit',
            'model': 'FREEDOM',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': None,
                'audio_feature_file': None,
            }
        },
        {
            'name': 'FREEDOM_text_minilm_image_vit',
            'model': 'FREEDOM',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': None,
            }
        },
        {
            'name': 'FREEDOM_text_clip_image_clip',
            'model': 'FREEDOM',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_clip.npy',
                'text_feature_file': 'text_clip.npy',
                'audio_feature_file': None,
            }
        },
            
        {
            'name': 'FREEDOM_text_minilm_image_vit_audio_vggish',
            'model': 'FREEDOM',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 1,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': 'audio_vggish.npy',
            }
        },
"""


import torch
print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")

def main():
    experiments = [
           {
            'name': 'FREEDOM_audioclip_full',
            'model': 'FREEDOM',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 1,
                'vision_feature_file': 'image_audioclip.npy',
                'text_feature_file': 'text_audioclip.npy',
                'audio_feature_file': 'audio_audioclip.npy',
            }
        },
    ]

    
    total_experiments = len(experiments)
    

    
    for idx, exp in enumerate(experiments, 1):
        try:
            exp['config']['experiment_name'] = exp.get('name', None)
            quick_start(
                model=exp['model'],
                dataset=exp['dataset'],
                config_dict=exp['config'],
                save_model=True
            )
            
        except Exception as e:
            print(f"\n Error in experiment {exp.get('name', 'N/A')}: {str(e)}")
            continue
    

if __name__ == '__main__':
    main()
