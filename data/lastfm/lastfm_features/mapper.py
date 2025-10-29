import pandas as pd
import numpy as np
import os

# Directory di output: parent della directory corrente
OUTPUT_DIR = os.path.dirname(os.getcwd())
print(f"Directory output: {OUTPUT_DIR}\n")

# ============================================
# STEP 1: Carica il file .inter e user_artists.dat
# ============================================
inter_df = pd.read_csv('lastfm.inter', sep='\t')
user_artists = pd.read_csv('user_artists.dat', sep='\t')

print(f"Dataset originale:")
print(f"  - Interazioni: {len(inter_df)}")
print(f"  - Users: {inter_df['userID'].nunique()}")
print(f"  - Items (remappati): {inter_df['itemID'].nunique()}")
print(f"  - Artists totali in user_artists.dat: {user_artists['artistID'].nunique()}")

# ============================================
# STEP 2: Crea mappatura inter_itemID -> original_artistID
# ============================================
# Gli artistID nel user_artists.dat sono gli ID originali
# Nel .inter, gli itemID sono remappati sequenzialmente
all_artists_sorted = sorted(user_artists['artistID'].unique())
inter_to_artist_map = {i: all_artists_sorted[i] for i in range(len(sorted(inter_df['itemID'].unique())))}

# Aggiungi la colonna artistID al dataframe delle interazioni
inter_df['artistID'] = inter_df['itemID'].map(inter_to_artist_map)

print(f"\nMappatura itemID -> artistID creata")
print(f"  - Items remappati: {len(inter_to_artist_map)}")

# ============================================
# STEP 3: Carica la mappatura item_id -> idx e filtra artisti disponibili
# ============================================
item_mapping = pd.read_csv('item_features.csv')
print(f"\nMappatura item_features caricata: {len(item_mapping)} righe")

# Estrai l'artistID da ogni item_id (es: "1000_1" -> 1000)
item_mapping['artistID'] = item_mapping['item_id'].str.split('_').str[0].astype(int)
item_mapping['variant'] = item_mapping['item_id'].str.split('_').str[1].astype(int)

# Identifica gli artisti con features multimodali disponibili
available_artists = set(item_mapping['artistID'].unique())
print(f"  - Artisti con features: {len(available_artists)}")
print(f"  - Varianti per artista: {item_mapping.groupby('artistID').size().value_counts().to_dict()}")

# ============================================
# STEP 4: Carica TUTTI i file .npy degli embeddings
# ============================================
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

print(f"\n✓ Caricati {len(embeddings)} tipi di embeddings, tutti con {num_items} righe (varianti)")

# ============================================
# STEP 5: Filtra le interazioni per artisti con features
# ============================================
valid_artists = available_artists
print(f"\nArtisti con features multimodali: {len(valid_artists)}")

# Filtra il dataset per includere solo artisti con features
filtered_df = inter_df[inter_df['artistID'].isin(valid_artists)].copy()
print(f"\nDopo filtro multimodale:")
print(f"  - Interazioni: {len(filtered_df)}")
print(f"  - Users: {filtered_df['userID'].nunique()}")
print(f"  - Artists: {filtered_df['artistID'].nunique()}")

# ============================================
# STEP 6: Applica 5-core filtering sugli ARTISTI
# ============================================
def core_k_filtering(interactions, k=5):
    """
    Perform Core-k filtering on a user-artist DataFrame.
    Ensures that every user and artist has at least k interactions.
    IMPORTANTE: Filtriamo sugli artistID, non sugli itemID remappati!
    """
    print(f"\nApplicazione {k}-core filtering (su artisti)...")
    iteration = 0
    while True:
        iteration += 1
        user_counts = interactions['userID'].value_counts()
        artist_counts = interactions['artistID'].value_counts()
        
        valid_users = user_counts[user_counts >= k].index
        valid_artists = artist_counts[artist_counts >= k].index
        
        core_k = interactions[
            interactions['userID'].isin(valid_users) & 
            interactions['artistID'].isin(valid_artists)
        ]
        
        print(f"  Iterazione {iteration}: {len(core_k)} interazioni, "
              f"{core_k['userID'].nunique()} users, {core_k['artistID'].nunique()} artists")
        
        if len(core_k) == len(interactions):
            break
        
        interactions = core_k
    
    return core_k

filtered_df = core_k_filtering(filtered_df, k=5)

print(f"\nDopo 5-core filtering:")
print(f"  - Interazioni: {len(filtered_df)}")
print(f"  - Users: {filtered_df['userID'].nunique()}")
print(f"  - Artists: {filtered_df['artistID'].nunique()}")

# ============================================
# STEP 7: Crea le mappature user e artist da 0 a m-1/n-1
# ============================================
unique_users = sorted(filtered_df['userID'].unique())
unique_artists = sorted(filtered_df['artistID'].unique())

map_user = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

# IMPORTANTE: Ora mappiamo TUTTE le varianti di ogni artista
# Ogni artista avrà N item (dove N = numero di varianti)
map_variant_to_item = {}  # (artistID, variant) -> nuovo itemID
inverse_map_item_to_variant = {}  # nuovo itemID -> (artistID, variant)

current_item_id = 0
for artist_id in unique_artists:
    # Trova tutte le varianti di questo artista
    artist_variants = item_mapping[item_mapping['artistID'] == artist_id].sort_values('variant')
    
    for _, row in artist_variants.iterrows():
        variant = row['variant']
        map_variant_to_item[(artist_id, variant)] = current_item_id
        inverse_map_item_to_variant[current_item_id] = (artist_id, variant, row['idx'])
        current_item_id += 1

print(f"\nUser remapping: 0 to {len(map_user)-1}")
print(f"Item remapping: 0 to {current_item_id-1} (include tutte le varianti)")
print(f"  - Artisti unici: {len(unique_artists)}")
print(f"  - Items totali (varianti): {current_item_id}")

# PROBLEMA: Nel dataset .inter abbiamo solo artistID, non sappiamo quale variante!
# SOLUZIONE: Espandiamo ogni interazione user-artist in N interazioni user-item
# dove N = numero di varianti dell'artista
print(f"\n⚠ ESPANSIONE DATASET: ogni interazione user-artist diventa N interazioni user-item")

expanded_rows = []
for _, row in filtered_df.iterrows():
    user_id = row['userID']
    artist_id = row['artistID']
    rating = row['rating']
    x_label = row.get('x_label', None)
    
    # Trova tutte le varianti di questo artista
    artist_variants = item_mapping[item_mapping['artistID'] == artist_id].sort_values('variant')
    
    # Crea una riga per ogni variante
    for _, variant_row in artist_variants.iterrows():
        variant = variant_row['variant']
        new_user_id = map_user[user_id]
        new_item_id = map_variant_to_item[(artist_id, variant)]
        
        if x_label is not None:
            expanded_rows.append({
                'userID': new_user_id,
                'itemID': new_item_id,
                'rating': rating,
                'x_label': x_label
            })
        else:
            expanded_rows.append({
                'userID': new_user_id,
                'itemID': new_item_id,
                'rating': rating
            })

final_df = pd.DataFrame(expanded_rows)

# ============================================
# STEP 8: Salva il nuovo file .inter nella directory parent
# ============================================
inter_output_path = os.path.join(OUTPUT_DIR, 'lastfm.inter')
final_df.to_csv(inter_output_path, sep='\t', index=False)
print(f"\n✓ Salvato: {inter_output_path}")

# ============================================
# STEP 9: Ricostruisci TUTTI i file .npy
# ============================================
print(f"\nRicostruzione file .npy...")
print(f"  Gli embeddings manterranno TUTTE le varianti di ogni artista")

# Dizionario per i nuovi embedding array
new_embeddings = {name: [] for name in embeddings.keys()}

# Per ogni nuovo itemID (0 to n-1), recupera il corrispondente embedding
for new_item_id in sorted(inverse_map_item_to_variant.keys()):
    artist_id, variant, original_idx = inverse_map_item_to_variant[new_item_id]
    
    # Recupera i feature vectors per tutti i tipi di embedding usando l'idx originale
    for name in embeddings.keys():
        new_embeddings[name].append(embeddings[name][original_idx])

# Converti in numpy array e salva nella directory parent
print(f"\nSalvataggio nuovi file embeddings nella directory parent:")
for name in embeddings.keys():
    new_array = np.array(new_embeddings[name])
    output_filename = f"{name}.npy"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    np.save(output_path, new_array)
    print(f"  ✓ {output_filename:30s}: {new_array.shape}")

print(f"\n✓ Tutti gli embeddings sono stati rimappati e salvati")

# ============================================
# STEP 10: Salva le nuove mappature nella directory parent
# ============================================
# Mappatura itemID -> (artistID, variant) 
new_item_features = pd.DataFrame([
    {
        'item_id': f"{artist_id}_{variant}", 
        'idx': item_id,
        'artist_id': artist_id,
        'variant': variant
    } 
    for item_id, (artist_id, variant, _) in inverse_map_item_to_variant.items()
])
item_features_path = os.path.join(OUTPUT_DIR, 'item_features.csv')
new_item_features.to_csv(item_features_path, index=False)

# Mappature di reference
user_mapping_df = pd.DataFrame([
    {'old_userID': old_id, 'new_userID': new_id} 
    for old_id, new_id in map_user.items()
])
user_mapping_path = os.path.join(OUTPUT_DIR, 'user_mapping.csv')
user_mapping_df.to_csv(user_mapping_path, index=False)

# Mappatura completa item
item_mapping_df = pd.DataFrame([
    {
        'new_itemID': item_id,
        'artistID': artist_id,
        'variant': variant,
        'original_idx': orig_idx
    } 
    for item_id, (artist_id, variant, orig_idx) in inverse_map_item_to_variant.items()
])
item_mapping_path = os.path.join(OUTPUT_DIR, 'item_mapping.csv')
item_mapping_df.to_csv(item_mapping_path, index=False)

print(f"\n✓ Salvati nella parent:")
print(f"  - item_features.csv (nuovo itemID -> artistID + variant)")
print(f"  - user_mapping.csv") 
print(f"  - item_mapping.csv")

# ============================================
# STEP 11: Statistiche finali
# ============================================
print(f"\n{'='*60}")
print(f"STATISTICHE FINALI")
print(f"{'='*60}")
print(f"Interazioni: {len(final_df)}")
print(f"Users: {final_df['userID'].nunique()} (da 0 a {final_df['userID'].max()})")
print(f"Items: {final_df['itemID'].nunique()} (da 0 a {final_df['itemID'].max()})")
print(f"Artists unici: {len(unique_artists)}")
print(f"Varianti medie per artista: {final_df['itemID'].nunique() / len(unique_artists):.1f}")

print(f"\n📁 File salvati nella parent directory ({OUTPUT_DIR}):")
print(f"  Dataset:")
print(f"    - lastfm.inter (itemID = tutte le varianti degli artisti)")
print(f"  Mappature:")
print(f"    - item_features.csv (itemID -> artistID + variant)")
print(f"    - user_mapping.csv (old -> new user IDs)")
print(f"    - item_mapping.csv (itemID -> artist info completo)")
print(f"  Embeddings:")
for name in embeddings.keys():
    print(f"    - {name}.npy")
print(f"{'='*60}")
print(f"\n✓ Nessun file sovrascritto nella directory corrente (lastfm_features)")
print(f"  NOTA: Nel nuovo dataset, ogni itemID rappresenta una specifica canzone/variante")
print(f"  NOTA: Gli embeddings mantengono TUTTE le varianti di ogni artista")
print(f"{'='*60}")