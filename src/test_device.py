#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test per verificare quale device viene effettivamente utilizzato
"""

import torch
import warnings

print("=" * 80)
print("TEST DEVICE PYTORCH")
print("=" * 80)

# Informazioni CUDA
print(f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"torch.cuda.get_device_capability(0): {torch.cuda.get_device_capability(0)}")

# Crea un device come fa il configurator
use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
print(f"\nDevice selezionato: {device}")

# Test reale: crea un tensore e vedi dove finisce
print("\n" + "=" * 80)
print("TEST ALLOCAZIONE TENSORE")
print("=" * 80)

try:
    # Crea un tensore sul device selezionato
    test_tensor = torch.randn(100, 100).to(device)
    print(f"\nTensore creato con successo")
    print(f"Tensore device: {test_tensor.device}")
    print(f"Tensore dtype: {test_tensor.dtype}")
    
    # Prova un'operazione
    result = test_tensor @ test_tensor.T
    print(f"\nMoltiplicazione matrici completata")
    print(f"Risultato device: {result.device}")
    
    # Verifica se è effettivamente sulla GPU
    if result.is_cuda:
        print(f"\n✓ Il tensore È effettivamente sulla GPU")
    else:
        print(f"\n✗ Il tensore NON è sulla GPU (è su CPU)")
        
except Exception as e:
    print(f"\n✗ ERRORE durante l'allocazione/operazione: {e}")
    print(f"   Il device fallback è: cpu")

print("\n" + "=" * 80)
