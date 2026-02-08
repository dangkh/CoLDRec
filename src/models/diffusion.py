import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings for diffusion process"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalUNet(nn.Module):
    """
    Simple U-Net style model for conditional diffusion
    Uses temb (text embedding) as guidance signal
    """
    def __init__(self, emb_dim, time_emb_dim=16, hidden_dim=64, text_emb_dim=None):
        super().__init__()
        
        if text_emb_dim is None:
            text_emb_dim = emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.LayerNorm(time_emb_dim),
            nn.GELU()
        )
        
        # Text conditioning projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Down blocks (encoder)
        self.down1 = nn.Sequential(
            nn.Linear(emb_dim + hidden_dim, hidden_dim),  # Input + text
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.down2 = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim + hidden_dim, hidden_dim * 2),  # h1 + time + text
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )
        
        # Bottleneck (MUST have text conditioning)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2),  # h2 + text ✓
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )
        
        # Up blocks (decoder) - CORRECTED: Now with text conditioning!
        self.up1 = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim * 2 + hidden_dim, hidden_dim),  # bottleneck + h2 + text ✓
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.up2 = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + hidden_dim, hidden_dim),  # up1 + h1 + text ✓
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Output
        self.out = nn.Linear(hidden_dim, emb_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, t, text_emb):
        """
        Args:
            x: noisy embedding at timestep t, shape [batch_size, emb_dim]
            t: timestep, shape [batch_size]
            text_emb: text embedding for guidance, shape [batch_size, text_emb_dim]
        """
        # Get embeddings
        t_emb = self.time_mlp(t)
        text_cond = self.text_proj(text_emb)
        
        # ============================================
        # ENCODER (Downsampling)
        # ============================================
        h1 = self.down1(torch.cat([x, text_cond], dim=-1))
        h2 = self.down2(torch.cat([h1, t_emb, text_cond], dim=-1))
        
        # ============================================
        # BOTTLENECK (with text conditioning)
        # ============================================
        h = self.bottleneck(torch.cat([h2, text_cond], dim=-1))
        
        # ============================================
        # DECODER (Upsampling) - NOW WITH TEXT CONDITIONING!
        # ============================================
        h = self.up1(torch.cat([h, h2, text_cond], dim=-1))  # ✓ Added text_cond
        h = self.up2(torch.cat([h, h1, text_cond], dim=-1))  # ✓ Added text_cond
        
        # Output
        return self.out(h)
    
def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

class ConditionalDDPM:
    """
    Conditional Denoising Diffusion Probabilistic Model
    """
    def __init__(self, model, T=10, beta_start=1e-4, beta_end=0.002, device='cuda', noiseScale = 1.0, schedule='linear'):
        self.model = model
        self.T = T
        self.device = device
        self.noiseScale = noiseScale
        
        # Linear schedule for beta
        if schedule in ['linear', 'linear_var']:
            start = self.noiseScale * beta_start
            end = self.noiseScale * beta_end
            self.betas = torch.linspace(start, end, T).to(device)
            if schedule == 'linear_var':
                self.betas = torch.tensor(betas_from_linear_variance(T, np.linspace(start, end, T)), dtype=torch.float32).to(device)
        else:
           raise ValueError(f"Unsupported noise schedule: {schedule}")
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to x_0 to get x_t
        
        Args:
            x_0: original embedding (cid), shape [batch_size, emb_dim]
            t: timestep, shape [batch_size]
            noise: optional noise tensor
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x_t, t, text_emb, guidance_scale=1.0, sample_noise=True):
        """
        Reverse diffusion process: denoise x_t to get x_{t-1}
        
        Args:
            x_t: noisy embedding at timestep t
            t: current timestep
            text_emb: text embedding for guidance (temb)
            guidance_scale: strength of guidance (1.0 = no guidance, >1.0 = stronger guidance)
        """
        batch_size = x_t.shape[0]
        
        # Predict noise with conditioning
        predicted_noise = self.model(x_t, t, text_emb)
        
        # Optional: Classifier-free guidance
        if guidance_scale != 1.0:
            # Predict unconditional noise (with zero text embedding)
            uncond_noise = self.model(x_t, t, torch.zeros_like(text_emb))
            # Apply guidance
            predicted_noise = uncond_noise + guidance_scale * (predicted_noise - uncond_noise)
        
        # Extract coefficients
        betas_t = self.betas[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None]
        
        # Compute mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if sample_noise is False or t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, cid, text_emb, infer_step, shape, guidance_scale=1.0):
        """
        Generate samples using reverse diffusion process
        
        Args:
            text_emb: text embedding for guidance (temb), shape [batch_size, text_emb_dim]
            shape: shape of embedding to generate, e.g., (batch_size, emb_dim)
            guidance_scale: guidance strength
        
        Returns:
            Generated embedding (denoised cid)
        """
        batch_size = shape[0]
        
        # Start from pure noise
        if infer_step == 0:
            x_t = cid
        else:
            t = torch.tensor([infer_step - 1] * cid.shape[0]).to(self.device)
            noise = torch.randn_like(cid)
            x_t = self.q_sample(cid, t, noise)
        
        # Reverse diffusion process
        for i in reversed(range(self.T)):
            t = torch.full((batch_size,), i, dtype=torch.long).to(self.device)
            x_t = self.p_sample(x_t, t, text_emb, guidance_scale)
        
        return x_t
    
    def train_diff(self, cid, temb):
        """
        Single training step
        
        Args:
            cid: target embedding to learn, shape [batch_size, emb_dim]
            temb: text embedding for conditioning, shape [batch_size, text_emb_dim]
        Returns:
            Loss value
        """
        batch_size = cid.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)
        
        # Sample noise
        noise = torch.randn_like(cid)
        
        # Forward diffusion: add noise to cid
        x_t = self.q_sample(cid, t, noise)
        
        # Predict noise using model
        predicted_noise = self.model(x_t, t, temb)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
        


# # Example usage
# if __name__ == "__main__":
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Hyperparameters
#     emb_dim = 64  # dimension of cid
#     text_emb_dim = 384  # dimension of temb (e.g., from BERT/GPT)
#     batch_size = 16
#     T = 5  # number of diffusion steps
    
#     # Initialize model
#     model = ConditionalUNet(
#         emb_dim=emb_dim,
#         time_emb_dim=128,
#         hidden_dim=256,
#         text_emb_dim=text_emb_dim
#     ).to(device)
    
#     # Initialize diffusion process
#     ddpm = ConditionalDDPM(model, T=T, device=device)
    
#     # Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
#     # Training example
#     print("Training example:")
#     for epoch in range(5):
#         # Dummy data
#         cid = torch.randn(batch_size, emb_dim).to(device)  # your embedding
#         temb = torch.randn(batch_size, text_emb_dim).to(device)  # text guidance
        
#         loss = ddpm.train_diff(cid, temb)
#         print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
#     # Sampling example
#     print("\nSampling example:")
#     model.eval()
#     test_temb = torch.randn(4, text_emb_dim).to(device)
#     generated_cid = ddpm.sample(
#         text_emb=test_temb,
#         shape=(4, emb_dim),
#         guidance_scale=2.0  # stronger guidance
#     )
#     print(f"Generated embedding shape: {generated_cid.shape}")
#     print(f"Generated embedding stats - Mean: {generated_cid.mean():.4f}, Std: {generated_cid.std():.4f}")