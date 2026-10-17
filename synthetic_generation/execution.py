"""Seeded CPU shuffle order with optional GPU row caching for diffusion."""
import torch
from torch.utils.data import DataLoader


class WindowBatches:
    """Keep the original CPU shuffle order; optionally gather windows on CUDA."""

    def __init__(self, windows, config):
        self.device = torch.device(config.device)
        self.last_batch_size = len(windows) % config.batch_size
        self.cached = False
        self.values = None
        if self.device.type == "cuda" and config.data_cache != "cpu":
            required = windows.values.size * 4
            free, _ = torch.cuda.mem_get_info(self.device)
            if required < free * 0.25:
                self.values = torch.as_tensor(windows.values, dtype=torch.float32,
                                              device=self.device)
                self.offsets = torch.arange(windows.length, device=self.device)
                self.cached = True
            elif config.data_cache == "cuda":
                raise ValueError("CUDA data cache would use over 25% of free memory; use auto or cpu.")
        self.loader = DataLoader(
            windows.starts if self.cached else windows,
            batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=torch.Generator().manual_seed(config.seed),
        )
        self.iterator = iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def next(self):
        try:
            value = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            value = next(self.iterator)
        value = value.to(self.device, non_blocking=self.device.type == "cuda")
        if self.cached:
            return self.values[value[:, None] + self.offsets[None, :]]
        return value

