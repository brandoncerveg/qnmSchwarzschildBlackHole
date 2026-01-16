"""
This code is based on https://github.com/AneleNcube/black-hole-quasinormal-modes-project-1/blob/001559e86311099bdadce9294e40e26489a73667/Eigenvalue%20Solver%20-%20Asymptotically%20Flat%20Schwarzschild%20BH.ipynb
whose author is Dr. Anele Ncube
"""

import os
import pickle
from datetime import datetime
from itertools import product
import argparse

import numpy as np
import matplotlib.pyplot as plt

from obtengo_fqn_base_optimized import (
    run_Scan_Black_Hole, 
    run_Scan_Black_Hole_Fast,
    QNMs, 
    plot_qnm_convergence, 
    plot_loss,
    DEVICE
)

# Configuración de matplotlib
FONT_SIZE = 15
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)


def main(fast_mode=False, use_amp=True):
    """
    Función principal para ejecutar el entrenamiento de QNM.
    
    Args:
        fast_mode: Si True, usa la versión ultra-rápida con menos logging
        use_amp: Si True, usa Automatic Mixed Precision (solo GPU)
    """
    
    print("\n" + "="*70)
    print(" "*15 + "BÚSQUEDA DE MODOS CUASINORMALES")
    print(" "*20 + "(VERSIÓN OPTIMIZADA)")
    print("="*70 + "\n")
    print(f"Dispositivo: {DEVICE}")
    print(f"Modo rápido: {'Sí' if fast_mode else 'No'}")
    print(f"AMP: {'Sí' if use_amp else 'No'}\n")
    
    # =============================================
    # CONFIGURACIÓN DE HIPERPARÁMETROS
    # =============================================
    

    xi0 = 0
    xif = 1
    
   
    n_train = 100          
    neurons = 10           
    epochs = int(20e3)     
    lr = 5e-3              
    minibatch = 1          
    

    S_values = [2]         # Spin
    M_values = [1]         # Masa
    L_values = [4]         # Momento angular
    N_values = [0]         # Overtone
    

    base_dir = os.path.dirname(os.path.abspath(__file__))
    historial_dir = os.path.join(base_dir, 'resultados')
    os.makedirs(historial_dir, exist_ok=True)
    

    timestamp = datetime.now().strftime("%d%m%y%H%M")

    hyperparams = {
        'n_train': n_train,
        'neurons': neurons,
        'epochs': epochs,
        'learning_rate': lr,
        'minibatch': minibatch,
        'xi0': xi0,
        'xif': xif,
        'optimizer': 'Adam',
        'activation': 'Sin',
        'device': str(DEVICE),
        'fast_mode': fast_mode,
        'amp': use_amp
    }
    
    print(f"Configuración:")
    print(f"  - Spin (S): {S_values}")
    print(f"  - Masa (M): {M_values}")
    print(f"  - Momento angular (L): {L_values}")
    print(f"  - Overtone (N): {N_values}")
    print(f"  - Puntos de entrenamiento: {n_train}")
    print(f"  - Neuronas por capa: {neurons}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"\nTimestamp: {timestamp}\n")
    

    
    results = []
    param_combinations = list(product(S_values, M_values, L_values, N_values))
    total_time = 0
    

    train_func = run_Scan_Black_Hole_Fast if fast_mode else run_Scan_Black_Hole
    
    for S, M, L, N in param_combinations:
        print(f"\n{'='*60}")
        print(f"Entrenando: S={S}, M={M}, L={L}, N={N}")
        print(f"{'='*60}")
        

        run_name = f"AFS_S={S},M={M},L={L},N={N}_{timestamp}"
        path = os.path.join(historial_dir, run_name)
        os.makedirs(path, exist_ok=True)
        

        hyperparams_run = hyperparams.copy()
        hyperparams_run.update({'S': S, 'M': M, 'L': L, 'N': N})
        

        if fast_mode:
            model, loss_hists, run_time = train_func(
                xi0, xif, neurons, epochs, n_train, lr, minibatch, S, M, L
            )
        else:
            model, loss_hists, run_time = train_func(
                xi0, xif, neurons, epochs, n_train, lr, minibatch, S, M, L,
                use_amp=use_amp
            )
        
        total_time += run_time
        
  
        omega1, omega2 = QNMs(loss_hists, path, run_time, hyperparams_run)
        

        results.append({
            "S": S,
            "M": M,
            "L": L,
            "N": N,
            "omega1": omega1,
            "omega2": omega2,
            "run_time": run_time,
            "modelo": model
        })
        

        print("\nGenerando visualizaciones...")
        plot_qnm_convergence(loss_hists, path, S, M, L, N)
        plot_loss(loss_hists, path, S, M, L, N, logscale=True)
        print("✓ Gráficas guardadas (3 PDFs)")
        

        pickle_run_path = os.path.join(path, "run_data.pkl")
        with open(pickle_run_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'loss_hists': loss_hists,
                'run_time': run_time,
                'omega1': omega1,
                'omega2': omega2,
                'hyperparams': hyperparams_run
            }, f)
    

    
    results_path = os.path.join(historial_dir, f'resultados_{timestamp}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*60}")
    print(f"✅ Resultados guardados en: {results_path}")
    print(f"⏱️  Tiempo total: {total_time/60:.2f} minutos")
    print(f"{'='*60}")
    
    # Resumen final
    print("\n--- RESUMEN DE RESULTADOS ---")
    for r in results:
        print(f"S={r['S']}, M={r['M']}, L={r['L']}: w = {r['omega1']:.6f} + {r['omega2']:.6f}i")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QNM PINN - Versión Optimizada')
    parser.add_argument('--fast', action='store_true', help='Usar modo ultra-rápido')
    parser.add_argument('--no-amp', action='store_true', help='Desactivar AMP')
    args = parser.parse_args()
    
    main(fast_mode=args.fast, use_amp=not args.no_amp)
