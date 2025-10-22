import pandas as pd
import numpy as np
import os

# Directory di output: parent della directory corrente
OUTPUT_DIR = os.path.dirname(os.getcwd())
print(f"Directory output: {OUTPUT_DIR}\n")

# ============================================
# STEP 1: Carica il file .inter
# ============================================
inter_df = pd.read_csv('movielens_1m.inter', sep='\t')
print(f"Dataset originale:")
print(f"  - Interazioni: {len(inter_df)}")
print(f"  - Users: {inter_df['userID'].nunique()}")
print(f"  - Items: {inter_df['itemID'].nunique()}")

# ============================================
# STEP 2: Carica la mappatura item_id -> idx
# ============================================
item_mapping = pd.read_csv('item_features.csv')
print(f"\nMappatura caricata: {len(item_mapping)} items")

# ============================================
# STEP 3: Carica TUTTI i file .npy degli embeddings
# ============================================
# Definisci tutti i file da caricare con nomi descrittivi
embedding_files = {
    # AudioCLIP (3 modalità)
    'image_audioclip': 'image_audioclip.npy',
    'audio_audioclip': 'audio_audioclip.npy',
    'text_audioclip': 'text_audioclip.npy',
    
    # CLIP (2 modalità)
    'image_clip': 'image_clip.npy',
    'text_clip': 'text_clip.npy',
    
    # MiniLM (1 modalità)
    'text_minilm': 'text_minilm.npy',
    
    # Whisper (1 modalità)
    'audio_vggish': 'audio_vggish.npy',
    
    # ViT (1 modalità)
    'image_vit': 'image_vit.npy',
}

# Carica tutti i file disponibili
embeddings = {}
num_items = None

print(f"\nCaricamento embeddings:")
for name, filename in embedding_files.items():
    try:
        embeddings[name] = np.load(filename)
        print(f"  ✓ {name:20s}: {embeddings[name].shape}")
        
        # Verifica coerenza del numero di item
        if num_items is None:
            num_items = embeddings[name].shape[0]
        else:
            assert embeddings[name].shape[0] == num_items, \
                f"Errore: {filename} ha {embeddings[name].shape[0]} item, atteso {num_items}!"
    except FileNotFoundError:
        print(f"  ⚠ {name:20s}: file non trovato, skip")

# Verifica che almeno un file sia stato caricato
if not embeddings:
    raise FileNotFoundError("Nessun file di embedding trovato!")

print(f"\n✓ Caricati {len(embeddings)} tipi di embeddings, tutti con {num_items} item")

# Tutti gli item nella mappatura hanno le 3 modalità
valid_items = set(item_mapping['item_id'])
print(f"\nItem con tutte e 3 le modalità (audio, immagini, testo): {len(valid_items)}")

# ============================================
# STEP 4: Filtra il dataset per includere solo valid_items
# ============================================
filtered_df = inter_df[inter_df['itemID'].isin(valid_items)].copy()
print(f"\nDopo filtro multimodale:")
print(f"  - Interazioni: {len(filtered_df)}")
print(f"  - Users: {filtered_df['userID'].nunique()}")
print(f"  - Items: {filtered_df['itemID'].nunique()}")

# ============================================
# STEP 5: Applica 5-core filtering
# ============================================
def core_k_filtering(interactions, k=5):
    """
    Perform Core-k filtering on a user-item DataFrame.
    Ensures that every user and item has at least k interactions.
    """
    print(f"\nApplicazione {k}-core filtering...")
    iteration = 0
    while True:
        iteration += 1
        user_counts = interactions['userID'].value_counts()
        item_counts = interactions['itemID'].value_counts()
        
        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index
        
        core_k = interactions[
            interactions['userID'].isin(valid_users) & 
            interactions['itemID'].isin(valid_items)
        ]
        
        print(f"  Iterazione {iteration}: {len(core_k)} interazioni, "
              f"{core_k['userID'].nunique()} users, {core_k['itemID'].nunique()} items")
        
        if len(core_k) == len(interactions):
            break
        
        interactions = core_k
    
    return core_k

filtered_df = core_k_filtering(filtered_df, k=5)

print(f"\nDopo 5-core filtering:")
print(f"  - Interazioni: {len(filtered_df)}")
print(f"  - Users: {filtered_df['userID'].nunique()}")
print(f"  - Items: {filtered_df['itemID'].nunique()}")

# ============================================
# STEP 6: Crea le mappature user e item da 0 a m-1/n-1
# ============================================
unique_users = sorted(filtered_df['userID'].unique())
unique_items = sorted(filtered_df['itemID'].unique())

map_user = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
map_item = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

inverse_map_user = {new_id: old_id for old_id, new_id in map_user.items()}
inverse_map_item = {new_id: old_id for old_id, new_id in map_item.items()}

print(f"\nUser remapping: 0 to {len(map_user)-1}")
print(f"Item remapping: 0 to {len(map_item)-1}")

# Applica il remapping
filtered_df['userID_new'] = filtered_df['userID'].map(map_user)
filtered_df['itemID_new'] = filtered_df['itemID'].map(map_item)

# Riordina le colonne
if 'x_label' in filtered_df.columns:
    final_df = filtered_df[['userID_new', 'itemID_new', 'rating', 'timestamp', 'x_label']]
    final_df.columns = ['userID', 'itemID', 'rating', 'timestamp', 'x_label']
else:
    final_df = filtered_df[['userID_new', 'itemID_new', 'rating', 'timestamp']]
    final_df.columns = ['userID', 'itemID', 'rating', 'timestamp']

# ============================================
# STEP 7: Salva il nuovo file .inter nella directory parent
# ============================================
inter_output_path = os.path.join(OUTPUT_DIR, 'movielens_1m.inter')
final_df.to_csv(inter_output_path, sep='\t', index=False)
print(f"\n✓ Salvato: {inter_output_path}")

# ============================================
# STEP 8: Ricostruisci TUTTI i file .npy
# ============================================
print(f"\nRicostruzione file .npy...")

# Crea la mappatura: vecchio_idx -> item_id
old_idx_to_item = dict(zip(item_mapping['idx'], item_mapping['item_id']))

# Dizionario per i nuovi embedding array
new_embeddings = {name: [] for name in embeddings.keys()}

# Per ogni nuovo indice (0 to n-1), trova il vecchio idx e recupera i feature vectors
for new_idx in range(len(inverse_map_item)):
    old_item_id = inverse_map_item[new_idx]
    
    # Trova il vecchio idx nel file .npy originale
    old_idx = item_mapping[item_mapping['item_id'] == old_item_id]['idx'].values[0]
    
    # Recupera i feature vectors per tutti i tipi di embedding
    for name in embeddings.keys():
        new_embeddings[name].append(embeddings[name][old_idx])

# Converti in numpy array e salva nella directory parent
print(f"\nSalvataggio nuovi file embeddings nella directory parent:")
for name in embeddings.keys():
    new_array = np.array(new_embeddings[name])
    output_filename = f"{name}.npy"  # Nome senza _filtered
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    np.save(output_path, new_array)
    print(f"  ✓ {output_filename:30s}: {new_array.shape}")

print(f"\n✓ Tutti gli embeddings sono stati rimappati e salvati")

# ============================================
# STEP 9: Salva le nuove mappature nella directory parent
# ============================================
# Mappatura item_id -> idx (nella parent - usata per caricare embeddings)
new_item_features = pd.DataFrame([
    {'item_id': old_id, 'idx': new_id} 
    for old_id, new_id in map_item.items()
])
item_features_path = os.path.join(OUTPUT_DIR, 'item_features.csv')
new_item_features.to_csv(item_features_path, index=False)

# Mappature di reference (anche queste nella parent per non sovrascrivere nulla)
user_mapping_df = pd.DataFrame([
    {'old_userID': old_id, 'new_userID': new_id} 
    for old_id, new_id in map_user.items()
])
user_mapping_path = os.path.join(OUTPUT_DIR, 'user_mapping.csv')
user_mapping_df.to_csv(user_mapping_path, index=False)

item_mapping_df = pd.DataFrame([
    {'old_itemID': old_id, 'new_itemID': new_id} 
    for old_id, new_id in map_item.items()
])
item_mapping_path = os.path.join(OUTPUT_DIR, 'item_mapping.csv')
item_mapping_df.to_csv(item_mapping_path, index=False)

print(f"\n✓ Salvati nella parent:")
print(f"  - item_features.csv")
print(f"  - user_mapping.csv") 
print(f"  - item_mapping.csv")

# ============================================
# STEP 10: Statistiche finali
# ============================================
print(f"\n{'='*60}")
print(f"STATISTICHE FINALI")
print(f"{'='*60}")
print(f"Interazioni: {len(final_df)}")
print(f"Users: {final_df['userID'].nunique()} (da 0 a {final_df['userID'].max()})")
print(f"Items: {final_df['itemID'].nunique()} (da 0 a {final_df['itemID'].max()})")

print(f"\n📁 File salvati nella parent directory ({OUTPUT_DIR}):")
print(f"  Dataset:")
print(f"    - movielens_1m.inter (per training)")
print(f"  Mappature:")
print(f"    - item_features.csv (item_id -> idx)")
print(f"    - user_mapping.csv (old -> new user IDs)")
print(f"    - item_mapping.csv (old -> new item IDs)")
print(f"  Embeddings:")
for name in embeddings.keys():
    print(f"    - {name}.npy")
print(f"{'='*60}")
print(f"\n✓ Nessun file sovrascritto nella directory corrente (features_mmrec)")
print(f"{'='*60}")