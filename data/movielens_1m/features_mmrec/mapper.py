import pandas as pd
import numpy as np

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
# STEP 3: Carica i file .npy - AUDIO, IMMAGINI, TESTO
# ============================================
audio_features = np.load('audios.npy')
image_features = np.load('images.npy')
text_features = np.load('texts.npy')

print(f"\nDimensioni features:")
print(f"  - Audio: {audio_features.shape}")
print(f"  - Immagini: {image_features.shape}")
print(f"  - Testo: {text_features.shape}")

# Verifica che tutti gli array abbiano lo stesso numero di item
assert audio_features.shape[0] == image_features.shape[0] == text_features.shape[0], \
    "I file .npy hanno dimensioni diverse!"

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
# STEP 7: Salva il nuovo file .inter
# ============================================
final_df.to_csv('movielens_1m_filtered.inter', sep='\t', index=False)
print(f"\n✓ Salvato: movielens_1m_filtered.inter")

# ============================================
# STEP 8: Ricostruisci i file .npy - AUDIO, IMMAGINI, TESTO
# ============================================
print(f"\nRicostruzione file .npy...")

# Crea la mappatura: vecchio_idx -> item_id
old_idx_to_item = dict(zip(item_mapping['idx'], item_mapping['item_id']))

# Liste per i nuovi array
new_audio = []
new_image = []
new_text = []

# Per ogni nuovo indice (0 to n-1), trova il vecchio idx e recupera il feature vector
for new_idx in range(len(inverse_map_item)):
    old_item_id = inverse_map_item[new_idx]
    
    # Trova il vecchio idx nel file .npy originale
    old_idx = item_mapping[item_mapping['item_id'] == old_item_id]['idx'].values[0]
    
    # Recupera i feature vectors
    new_audio.append(audio_features[old_idx])
    new_image.append(image_features[old_idx])
    new_text.append(text_features[old_idx])

# Converti in numpy array
new_audio_array = np.array(new_audio)
new_image_array = np.array(new_image)
new_text_array = np.array(new_text)

print(f"  - Nuove dimensioni Audio: {new_audio_array.shape}")
print(f"  - Nuove dimensioni Immagini: {new_image_array.shape}")
print(f"  - Nuove dimensioni Testo: {new_text_array.shape}")

# Salva i nuovi file .npy
np.save('audio_filtered.npy', new_audio_array)
np.save('image_filtered.npy', new_image_array)
np.save('text_filtered.npy', new_text_array)

print(f"\n✓ Salvati: audio_filtered.npy, image_filtered.npy, text_filtered.npy")

# ============================================
# STEP 9: Salva le nuove mappature
# ============================================
# Mappatura user
user_mapping_df = pd.DataFrame([
    {'old_userID': old_id, 'new_userID': new_id} 
    for old_id, new_id in map_user.items()
])
user_mapping_df.to_csv('user_mapping.csv', index=False)

# Mappatura item
item_mapping_df = pd.DataFrame([
    {'old_itemID': old_id, 'new_itemID': new_id} 
    for old_id, new_id in map_item.items()
])
item_mapping_df.to_csv('item_mapping.csv', index=False)

# Nuova mappatura item_id -> idx
new_item_features = pd.DataFrame([
    {'item_id': old_id, 'idx': new_id} 
    for old_id, new_id in map_item.items()
])
new_item_features.to_csv('item_features_filtered.csv', index=False)

print(f"\n✓ Salvati: user_mapping.csv, item_mapping.csv, item_features_filtered.csv")

# ============================================
# STEP 10: Statistiche finali
# ============================================
print(f"\n{'='*60}")
print(f"STATISTICHE FINALI")
print(f"{'='*60}")
print(f"Interazioni: {len(final_df)}")
print(f"Users: {final_df['userID'].nunique()} (da 0 a {final_df['userID'].max()})")
print(f"Items: {final_df['itemID'].nunique()} (da 0 a {final_df['itemID'].max()})")
print(f"\nModalità: AUDIO + IMMAGINI + TESTO")
print(f"\nFile generati:")
print(f"  - movielens_1m_filtered.inter")
print(f"  - audio_filtered.npy")
print(f"  - image_filtered.npy")
print(f"  - text_filtered.npy")
print(f"  - user_mapping.csv")
print(f"  - item_mapping.csv")
print(f"  - item_features_filtered.csv")
print(f"{'='*60}")