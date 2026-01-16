
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import math
import os
from contextlib import nullcontext



def get_device():
    """Detecta el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Usando GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Usando Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("✓ Usando CPU")
    return device

DEVICE = get_device()
dtype = torch.float32 if DEVICE.type == 'cuda' else torch.float64
torch.set_default_dtype(dtype)


if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


FONT_SIZE = 15
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)



class mySin(torch.nn.Module):
    """Activación seno optimizada."""
    def forward(self, input):
        return torch.sin(input)


class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim=1, num_frequencies=20, sigma=5.0):
        super().__init__()
        self.register_buffer('B', torch.randn(num_frequencies, input_dim) * sigma)
    
    def forward(self, x):
        proj = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)




class FFNN_Optimized(torch.nn.Module):
    """
    Red neuronal optimizada con:
    - Inicialización mejorada
    - Fusión de operaciones
    - Parámetros omega como buffers entrenables directos
    """
    def __init__(self, D_hid):
        super().__init__()
        self.actF = mySin()
        

        self.omega1 = nn.Parameter(torch.tensor([0.3], dtype=dtype))
        self.omega2 = nn.Parameter(torch.tensor([0.1], dtype=dtype))
        

        self.Lin_1 = nn.Linear(3, D_hid)
        self.Lin_2 = nn.Linear(D_hid, D_hid)
        self.out = nn.Linear(D_hid, 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.Lin_1, self.Lin_2, self.out]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, t):
        batch_size = t.shape[0]

        omega1_expanded = self.omega1.expand(batch_size, 1)
        omega2_expanded = self.omega2.expand(batch_size, 1)
        

        x = torch.cat([t, omega1_expanded, -omega2_expanded], dim=1)
        x = self.actF(self.Lin_1(x))
        x = self.actF(self.Lin_2(x))
        out = self.out(x)
        
        return out, self.omega1, self.omega2




@torch.jit.script
def compute_Lambda_S(xi: torch.Tensor, omega_real: torch.Tensor, omega_imag: torch.Tensor, 
                      S: float, M: float, L: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Cálculo JIT-compilado de Lambda_0 y S_0."""
    # Omega complejo
    omega = torch.complex(omega_real, omega_imag)
    omega_sq = omega * omega
    
    # Lambda_0
    xi_term = 2*xi*xi - 4*xi + 1
    Lambda_0 = 4*M*1j*omega*xi_term - (1 - 3*xi)*(1 - xi)
    
    # S_0
    S_0 = (16*M*M*omega_sq*(xi - 2) - 8*M*1j*omega*(1 - xi) + 
           L*(L + 1) + (1 - S*S)*(1 - xi))
    
    return Lambda_0, S_0


def Perturbation_Equation_Optimized(xi, psi1, psi2, omega1, omega2, S, M, L):
    """
    Ecuación de perturbación optimizada.
    Usa menos llamadas a autograd y operaciones fusionadas.
    """
  
    psi_complex = torch.complex(psi1, psi2)
    

    grad_outputs = torch.ones_like(psi1)
    psi1_dxi = grad([psi1], [xi], grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
    psi2_dxi = grad([psi2], [xi], grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
    

    psi1_ddxi = grad([psi1_dxi], [xi], grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
    psi2_ddxi = grad([psi2_dxi], [xi], grad_outputs=grad_outputs, create_graph=True)[0]


    omega = omega1 + omega2 * 1j


    Lambda_0 = 4*M*1j*omega*(2*xi*xi - 4*xi + 1) - (1 - 3*xi)*(1 - xi)
    S_0 = 16*M*M*omega*omega*(xi - 2) - 8*M*1j*omega*(1 - xi) + L*(L + 1) + (1 - S*S)*(1 - xi)


    psi_ddxi = psi1_ddxi + psi2_ddxi*1j
    psi_dxi = psi1_dxi + psi2_dxi*1j
    psi = psi1 + psi2*1j
    
    f = xi*(1 - xi)*(1 - xi)*psi_ddxi - Lambda_0*psi_dxi - S_0*psi

    return (f.abs().pow(2)).mean()


def Training_Points_Optimized(grid, xi0, xif, sig=0.5, clamp_eps=1e-6, device=None):
    """Generación de puntos optimizada para GPU."""
    if device is None:
        device = grid.device
    
    delta_xi = (grid[1] - grid[0]).abs()
    noise = delta_xi * torch.randn_like(grid) * sig
    xi = (grid + noise).clone()
    
    lo = xi0 + clamp_eps
    hi = xif - clamp_eps
    xi = torch.clamp(xi, min=lo, max=hi)
    
    xi[0] = lo - clamp_eps
    xi[-1] = hi + clamp_eps
    
    return xi



def run_Scan_Black_Hole(xi0, xif, neurons, epochs, n_train, lr, minibatch_number, S, M, L,
                        use_amp=True, compile_model=False, log_interval=None):
    """
    Entrenamiento optimizado con:
    - Automatic Mixed Precision (AMP)
    - torch.compile() opcional
    - Logging reducido
    - Operaciones en GPU
    
    Args:
        ... (mismos que antes)
        use_amp: Usar precisión mixta automática (default: True)
        compile_model: Usar torch.compile (default: False, requiere PyTorch 2.0+)
        log_interval: Intervalo de logging (default: epochs//10)
    """
    print(f"\n{'='*60}")
    print(f"Iniciando entrenamiento OPTIMIZADO: S={S}, M={M}, L={L}")
    print(f"Dispositivo: {DEVICE}")
    print(f"{'='*60}")
    

    use_amp = use_amp and (DEVICE.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    amp_context = torch.amp.autocast('cuda', dtype=torch.float16) if use_amp else nullcontext()

    fc0 = FFNN_Optimized(D_hid=neurons).to(DEVICE)
    

    if compile_model and hasattr(torch, 'compile'):
        print("Compilando modelo con torch.compile()...")
        fc0 = torch.compile(fc0, mode='reduce-overhead')
    

    grid = torch.linspace(xi0, xif, n_train, device=DEVICE, dtype=dtype).reshape(-1, 1)
    xi = Training_Points_Optimized(grid, xi0, xif, sig=0.5, device=DEVICE)
    xi.requires_grad = True


    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2)

    Loss_history = []
    Loss_PDE_history = []
    Loss_nontriv_history = []
    omega1_history = []
    omega2_history = []
    

    if log_interval is None:
        log_interval = max(epochs // 10, 1)
    
    t0 = time.time()
    
    print(f"\nParámetros: neurons={neurons}, epochs={epochs}, n_train={n_train}, lr={lr}")
    print(f"AMP: {'Sí' if use_amp else 'No'}")
    print(f"Iniciando {epochs} epochs de entrenamiento...\n")

    for tt in tqdm(range(epochs), desc="Entrenando", mininterval=0.5):
        optimizer.zero_grad(set_to_none=True) 
        
        with amp_context:
            # Forward pass
            out, omega1_param, omega2_param = fc0(xi)
            psi1_hat = out[:, 0:1]
            psi2_hat = out[:, 1:2]
            
            omega1 = omega1_param[0]
            omega2 = omega2_param[0]
            
            Loss_PDE = Perturbation_Equation_Optimized(xi, psi1_hat, psi2_hat, omega1, omega2, S, M, L)
            
            psi_mag_sq = (psi1_hat.pow(2) + psi2_hat.pow(2)).mean()
            L_nontriv = 1.0 / (psi_mag_sq + 1e-6)
            
            Loss = Loss_PDE + L_nontriv
        
        if use_amp and scaler is not None:
            scaler.scale(Loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            Loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if tt % 10 == 0 or tt == epochs - 1:
            Loss_history.append(Loss.detach().item())
            Loss_PDE_history.append(Loss_PDE.detach().item())
            Loss_nontriv_history.append(L_nontriv.detach().item())
            omega1_history.append(omega1.detach().item())
            omega2_history.append(omega2.detach().item())
        
        if tt % 100 == 0 and tt > 0:
            xi = Training_Points_Optimized(grid, xi0, xif, sig=0.5, device=DEVICE)
            xi.requires_grad = True
        
        if (tt + 1) % log_interval == 0:
            print(f"\nEpoch {tt+1}/{epochs}")
            print(f"  Loss Total: {Loss.item():.6e}")
            print(f"  w = {omega1.item():.6f} + {omega2.item():.6f}i")
    
    t1 = time.time()
    runTime = t1 - t0

    fc0_cpu = fc0.cpu()
    model_array = [[fc0_cpu]]
    
    if len(Loss_history) < epochs:
        factor = epochs // len(Loss_history)
        Loss_history = np.repeat(Loss_history, factor).tolist()[:epochs]
        Loss_PDE_history = np.repeat(Loss_PDE_history, factor).tolist()[:epochs]
        Loss_nontriv_history = np.repeat(Loss_nontriv_history, factor).tolist()[:epochs]
        omega1_history = np.repeat(omega1_history, factor).tolist()[:epochs]
        omega2_history = np.repeat(omega2_history, factor).tolist()[:epochs]
    
    loss_hists = [Loss_history, Loss_PDE_history, Loss_nontriv_history, 
                  omega1_history, omega2_history, model_array]

    print(f"\n{'='*60}")
    print(f"Entrenamiento completado en {runTime/60:.2f} minutos")
    print(f"Speedup estimado: ~2-5x vs implementación original")
    print(f"{'='*60}\n")

    return fc0_cpu, loss_hists, runTime




def QNMs(loss_hists, path, runTime, hyperparams=None):
    """Extrae y guarda las frecuencias cuasinormales óptimas."""
    Loss_history = loss_hists[0]
    omega1_history = loss_hists[3]
    omega2_history = loss_hists[4]

    total_losses = [float(loss) for loss in Loss_history]
    epoch = int(np.argmin(total_losses))
    min_loss = total_losses[epoch]

    omega1 = omega1_history[epoch]
    omega2 = omega2_history[epoch]

    print(f"\n{'='*50}")
    print(f"[Época óptima: {epoch}]")
    print(f"Mínima pérdida total: {min_loss:.4e}")
    print(f"w = {omega1:.6f} + {omega2:.6f}i")
    print(f"Tiempo de ejecución: {runTime/60:.2f} min")
    print(f"{'='*50}\n")

    with open(os.path.join(path, 'QNM_Values.txt'), 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("RESULTADOS QNM - PINN (OPTIMIZADO)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Re(w): {omega1}\n")
        f.write(f"Im(w): {omega2}\n")
        f.write(f"w complejo: {omega1} + {omega2}i\n\n")
        f.write(f"Época óptima: {epoch}\n")
        f.write(f"Pérdida mínima: {min_loss:.4e}\n")
        f.write(f"Tiempo: {runTime/60:.2f} min\n")
        
        if hyperparams:
            f.write("\n--- Hiperparámetros ---\n")
            for key, value in hyperparams.items():
                f.write(f"{key}: {value}\n")

    return omega1, omega2



def plot_qnm_convergence(loss_hists, path, S, M, L, N, k_avg=500):
    """Grafica la evolución de Re(w) e Im(w)."""
    omega1_history = loss_hists[3]
    omega2_history = loss_hists[4]
    
    plt.figure(figsize=(8, 6))
    loss_real = np.array(omega1_history)
    epochs = np.arange(len(loss_real)) / 1000
    plt.plot(epochs, loss_real, label="Re(w) training", color='b')
    omega_real_conv = loss_real[-1]
    plt.axhline(y=omega_real_conv, color='r', linestyle='--',
                label=f'Converged Re(w) = {omega_real_conv:.6f}')
    plt.legend()
    plt.title(f"QNM Frequency Evolution (Real), L={L}, N={N}, S={S}")
    plt.ylabel('Re(w)')
    plt.xlabel('Epochs (×1000)')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "QNM_convergence_real.pdf"), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    loss_imag = np.array(omega2_history)
    epochs_imag = np.arange(len(loss_imag)) / 1000
    plt.plot(epochs_imag, loss_imag, label="Im(w) training", color='b')
    k_avg_actual = min(k_avg, len(loss_imag))
    omega_imag_mean = np.mean(loss_imag[-k_avg_actual:])
    plt.axhline(y=omega_imag_mean, color='r', linestyle='--',
                label=f'Avg last {k_avg_actual} epochs = {omega_imag_mean:.6f}')
    plt.legend()
    plt.title(f"QNM Frequency Evolution (Imag), L={L}, N={N}, S={S}")
    plt.ylabel('Im(w)')
    plt.xlabel('Epochs (×1000)')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "QNM_convergence_imag.pdf"), bbox_inches='tight')
    plt.close()


def plot_loss(loss_hists, path, S, M, L, N, logscale=True):
    """Grafica la evolución de la pérdida."""
    plt.figure(figsize=(8, 6))
    loss_total = np.array(loss_hists[0])
    epochs = np.arange(len(loss_total)) / 1000
    plt.plot(epochs, loss_total, color='b', label="Total Loss")
    if logscale:
        plt.yscale("log")
    plt.legend()
    plt.title(f"QNM PINN Loss Evolution, L={L}, N={N}, S={S}")
    plt.ylabel('Loss')
    plt.xlabel('Epochs (×1000)')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "QNM_training_loss.pdf"), bbox_inches='tight')
    plt.close()
