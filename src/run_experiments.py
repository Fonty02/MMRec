# coding: utf-8
"""
Script per eseguire tutti gli esperimenti configurati
"""

import os
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'


def main():
    experiments = [
        {
            
            'name': 'LATTICE_text_minilm',
            'model': 'LATTICE',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': None,  # Non usare visual
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': None,  # Non usare audio
            }
        },
        {
            'name': 'LATTICE_audio_vggish',
            'model': 'LATTICE',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': None,
                'text_feature_file': None,
                'audio_feature_file': 'audio_vggish.npy',
            }
        },
        {
            'name': 'LATTICE_image_vit',
            'model': 'LATTICE',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': None,
                'audio_feature_file': None,
            }
        },
        {
            'name': 'LATTICE_text_minilm_image_vit',
            'model': 'LATTICE',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': None,
            }
        },
        {
            'name': 'LATTICE_text_clip_image_clip',
            'model': 'LATTICE',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_clip.npy',
                'text_feature_file': 'text_clip.npy',
                'audio_feature_file': None,
            }
        },
        {
            'name': 'LATTICE_text_minilm_image_vit_audio_vggish',
            'model': 'LATTICE',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': 'audio_vggish.npy',
            }
        },
        {
            'name': 'LATTICE_audioclip_full',
            'model': 'LATTICE',
            'dataset': 'movielens_1m',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_audioclip.npy',
                'text_feature_file': 'text_audioclip.npy',
                'audio_feature_file': 'audio_audioclip.npy',
            }
        },
        {
            
            'name': 'LATTICE_text_minilm',
            'model': 'LATTICE',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': None,  # Non usare visual
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': None,  # Non usare audio
            }
        },
        {
            'name': 'LATTICE_image_vit',
            'model': 'LATTICE',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': None,
                'audio_feature_file': None,
            }
        },
        {
            'name': 'LATTICE_text_minilm_image_vit',
            'model': 'LATTICE',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_vit.npy',
                'text_feature_file': 'text_minilm.npy',
                'audio_feature_file': None,
            }
        },
        {
            'name': 'LATTICE_text_clip_image_clip',
            'model': 'LATTICE',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_clip.npy',
                'text_feature_file': 'text_clip.npy',
                'audio_feature_file': None,
            }
        },

        {
            'name': 'LATTICE_audioclip_full',
            'model': 'LATTICE',
            'dataset': 'lastfm',
            'config': {
                'gpu_id': 0,
                'vision_feature_file': 'image_audioclip.npy',
                'text_feature_file': 'text_audioclip.npy',
                'audio_feature_file': 'audio_audioclip.npy',
            }
        },
    ]

    
    total_experiments = len(experiments)
    
    print("="*80)
    print(f"INIZIO ESECUZIONE {total_experiments} ESPERIMENTI")
    print("="*80)
    
    for idx, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"ESPERIMENTO {idx}/{total_experiments}: {exp['name']}")
        print(f"{'='*80}")
        print(f"Modello: {exp['model']}")
        print(f"Dataset: {exp['dataset']}")
        if exp['model'] == 'LATTICE':
            print(f"Features:")
            print(f"  - Vision: {exp['config'].get('vision_feature_file', 'None')}")
            print(f"  - Text:   {exp['config'].get('text_feature_file', 'None')}")
            print(f"  - Audio:  {exp['config'].get('audio_feature_file', 'None')}")
        print(f"{'='*80}\n")
        
        try:
            # Esegui l'esperimento
            quick_start(
                model=exp['model'],
                dataset=exp['dataset'],
                config_dict=exp['config'],
                save_model=True
            )
            print(f"\n✓ Esperimento {idx}/{total_experiments} completato con successo!")
            
        except Exception as e:
            print(f"\n✗ Errore nell'esperimento {idx}/{total_experiments}: {e}")
            print(f"   Continuo con il prossimo esperimento...")
            continue
    
    print("\n" + "="*80)
    print("TUTTI GLI ESPERIMENTI COMPLETATI!")
    print("="*80)
    print(f"\nRisultati salvati in: reports/best_results_*.csv")
    print("="*80)


if __name__ == '__main__':
    main()
