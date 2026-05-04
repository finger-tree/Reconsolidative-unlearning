import torch
import torch.nn as nn

class GridCellLayer(nn.Module):
    """Grid cell-like spatial encoding from entorhinal cortex"""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Different spatial frequencies (like biological grid cells)
        self.freq1 = nn.Conv2d(channels, channels//4, 1)
        self.freq2 = nn.Conv2d(channels, channels//4, 1)
        self.freq3 = nn.Conv2d(channels, channels//4, 1)
        self.freq4 = nn.Conv2d(channels, channels//4, 1)
        
        # Phase offsets (creates hexagonal patterns)
        self.register_buffer('phase1', torch.randn(1, channels//4, 1, 1))
        self.register_buffer('phase2', torch.randn(1, channels//4, 1, 1))
        self.register_buffer('phase3', torch.randn(1, channels//4, 1, 1))
        self.register_buffer('phase4', torch.randn(1, channels//4, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Different spatial frequencies with phase offsets
        y1 = torch.sin(self.freq1(x) + self.phase1)
        y2 = torch.sin(self.freq2(x) + self.phase2)
        y3 = torch.sin(self.freq3(x) + self.phase3)
        y4 = torch.sin(self.freq4(x) + self.phase4)
        return torch.cat([y1, y2, y3, y4], dim=1)
    

if __name__ == "__main__":
    # show  grid cell patterns (testing)
    import matplotlib.pyplot as plt
    grid_layer = GridCellLayer(64)
    input = torch.randn(1, 64, 32, 32)  # batch size 1, 64 channels, 32x32 spatial dimensions
    output = grid_layer(input)
    # Visualize the first few channels of the output
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        axes[i//4, i%4].imshow(output[0, i].detach().numpy(), cmap='viridis')
        axes[i//4, i%4].set_title(f'Channel {i}')
    plt.show()
