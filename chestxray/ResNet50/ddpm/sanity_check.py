import sys, torch
sys.path.append('/mnt/data1/gotou/projects/chestxray/ddpm')
import ddpm_train as m

m.model.eval()
B, C, H, W = 2, 1, m.IMAGE_SIZE, m.IMAGE_SIZE
x_t = torch.randn(B, C, H, W, device=m.DEVICE)
with torch.no_grad():
    t = torch.randint(0, m.TIMESTEPS, (B,), device=m.DEVICE).long()
    out = m.model(x_t, t)
    print('model_out', tuple(out.shape))
    x_prev = m.p_sample(m.model, x_t, t)
    print('p_sample_out', tuple(x_prev.shape))
print('OK')
