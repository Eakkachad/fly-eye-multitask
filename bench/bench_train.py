# Benchmark: flyvis connectome network forward+backward speed / VRAM on random hex movies.
import time, torch, flyvis
from flyvis import Network
torch.manual_seed(0)
net = Network()  # default connectome-constrained config
dev = next(net.parameters()).device
n_hex = net.connectome.input_list if hasattr(net.connectome,'input_list') else None
B, T = 4, 19
# stimulus: (batch, frames, 1, n_hexals) — infer hexals from network
from flyvis.utils.hex_utils import get_num_hexals
n = get_num_hexals(15)
dt = 1/50
opt = torch.optim.Adam(net.parameters(), lr=5e-5)
x = torch.rand(B, T, 1, n, device=dev)
ss = net.steady_state(t_pre=0.5, dt=dt, batch_size=B, value=0.5)
def step():
    opt.zero_grad()
    net.stimulus.zero(B, T); net.stimulus.add_input(x)
    act = net(net.stimulus(), dt, state=ss)
    loss = act.pow(2).mean()
    loss.backward(); opt.step()
for _ in range(2): step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t=time.time(); N=10
for _ in range(N): step()
torch.cuda.synchronize()
print(f"params={sum(p.numel() for p in net.parameters())}  s/iter={(time.time()-t)/N:.3f}  peakVRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GB  device={dev}")
