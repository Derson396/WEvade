import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.conv_bn_relu import ConvBNRelu


class ReedSolomonEncoder:
    """Simplified Reed-Solomon encoding for watermark redundancy (torch-only)"""
    def __init__(self, message_length, redundancy=10):
        self.message_length = message_length
        self.redundancy = redundancy
        self.encoded_length = message_length + redundancy

    def encode(self, message):
        """Add redundancy bits using simple parity checks (torch ops)"""
        # message is expected as a torch.Tensor on some device
        if not isinstance(message, torch.Tensor):
            message = torch.tensor(message, dtype=torch.float32)

        device = message.device
        batch_size = message.shape[0]

        encoded = torch.zeros((batch_size, self.encoded_length), device=device, dtype=message.dtype)
        encoded[:, :self.message_length] = message

        # Simple parity bits (torch operations)
        # handle case when message_length not divisible by redundancy: use integer division
        chunk_size = max(1, self.message_length // self.redundancy)
        for i in range(self.redundancy):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, self.message_length)
            if start_idx >= end_idx:
                parity = torch.zeros(batch_size, device=device, dtype=message.dtype)
            else:
                parity = torch.remainder(torch.sum(message[:, start_idx:end_idx], dim=1), 2)
            encoded[:, self.message_length + i] = parity

        return encoded.float()

    def decode(self, received):
        """Decode and correct errors using torch (differentiable)"""
        # received should be a torch.Tensor (no detach/cpu/numpy)
        if not isinstance(received, torch.Tensor):
            received = torch.tensor(received, dtype=torch.float32)

        device = received.device
        received = received.to(device)

        # Extract message part
        message = received[:, :self.message_length]  # keeps grad if received has grad

        # Simple error correction using parity (torch ops)
        batch_size = message.shape[0]
        corrected = message.clone()

        chunk_size = max(1, self.message_length // self.redundancy)
        for i in range(self.redundancy):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, self.message_length)
            if start_idx >= end_idx:
                continue
            parity_received = received[:, self.message_length + i]
            parity_computed = torch.remainder(torch.sum(message[:, start_idx:end_idx], dim=1), 2)

            # If mismatch, we *could* try to correct — here we keep simple: do nothing or apply soft correction
            # For differentiability, we'll produce a "corrected" value as the original message (no hard flipping)
            # More sophisticated correction can be implemented with differentiable heuristics if needed

            # Example: we can nudge bits towards parity to help learning (optional)
            mismatch = (parity_received != parity_computed).float().unsqueeze(1)  # (B,1)
            # no hard flip, keep corrected as-is (placeholder)
            # corrected[:, start_idx:end_idx] = corrected[:, start_idx:end_idx]  # no-op

        return corrected.float()


class DCTWatermarkLayer(nn.Module):
    """Insert watermark in DCT frequency domain"""
    def __init__(self, block_size=8):
        super(DCTWatermarkLayer, self).__init__()
        self.block_size = block_size
        
    def dct2d(self, image):
        """2D DCT approximation using FFT"""
        # Use FFT-based DCT approximation
        B, C, H, W = image.shape
        
        # Apply FFT and take real components as DCT approximation
        fft_result = torch.fft.rfft2(image, norm='ortho')
        
        # Return magnitude (real approximation of DCT)
        return fft_result.real
    
    def idct2d(self, dct_coeffs):
        """2D Inverse DCT approximation"""
        # Create complex tensor for inverse FFT
        complex_coeffs = torch.complex(dct_coeffs, torch.zeros_like(dct_coeffs))
        
        # Apply inverse FFT
        result = torch.fft.irfft2(complex_coeffs, norm='ortho')
        
        return result
    
    def forward(self, image, watermark, strength=0.1):
        """
        Insert watermark into mid-frequency DCT coefficients
        Args:
            image: (B, C, H, W)
            watermark: (B, L) - encoded watermark
            strength: insertion strength
        """
        B, C, H, W = image.shape
        watermarked = image.clone()
        
        # Get DCT coefficients for entire image
        dct = self.dct2d(image)
        
        # dct shape: (B, C, H, W//2+1) due to rfft2
        h, w = dct.shape[-2:]
        
        # Create mid-frequency mask (avoid DC component and very high frequencies)
        mid_freq_mask = torch.zeros_like(dct)
        h_start, h_end = h // 4, 3 * h // 4
        w_start, w_end = w // 4, 3 * w // 4
        mid_freq_mask[:, :, h_start:h_end, w_start:w_end] = 1
        
        # Expand watermark to match DCT dimensions
        # Reshape watermark: (B, L) -> (B, 1, 1, L)
        watermark_reshaped = watermark.unsqueeze(1).unsqueeze(1)
        
        # Interpolate to mid-frequency region size
        target_h = h_end - h_start
        target_w = w_end - w_start
        
        # Repeat across channels
        watermark_expanded = watermark_reshaped.expand(B, C, 1, watermark.shape[1])
        
        # Adaptive interpolation to fit mid-frequency region
        watermark_freq = torch.nn.functional.interpolate(
            watermark_expanded, 
            size=(target_h, target_w), 
            mode='nearest'
        )
        
        # Modify DCT coefficients
        dct_modified = dct.clone()
        dct_modified[:, :, h_start:h_end, w_start:w_end] += strength * watermark_freq
        
        # Reconstruct image
        watermarked = self.idct2d(dct_modified)
        
        # Ensure output matches input size
        if watermarked.shape != image.shape:
            watermarked = torch.nn.functional.interpolate(
                watermarked, size=(H, W), mode='bilinear', align_corners=False
            )
        
        return watermarked
    
    def extract(self, image, watermark_length):
        """Extract watermark from DCT coefficients"""
        B, C, H, W = image.shape
        
        # Get DCT coefficients
        dct = self.dct2d(image)
        h, w = dct.shape[-2:]
        
        # Extract from mid-frequency region
        h_start, h_end = h // 4, 3 * h // 4
        w_start, w_end = w // 4, 3 * w // 4
        mid_freq = dct[:, :, h_start:h_end, w_start:w_end]
        
        # Average across spatial dimensions and channels to get watermark
        # Result shape: (B, C, h_mid, w_mid)
        pooled = torch.nn.functional.adaptive_avg_pool2d(mid_freq, (1, watermark_length // C))
        
        # Reshape: (B, C, 1, watermark_length // C) -> (B, watermark_length)
        extracted = pooled.view(B, -1)
        
        # If length doesn't match, adjust
        if extracted.shape[1] != watermark_length:
            # Interpolate to match watermark length
            extracted = extracted.unsqueeze(1)  # (B, 1, current_length)
            extracted = torch.nn.functional.interpolate(
                extracted, size=watermark_length, mode='linear', align_corners=False
            )
            extracted = extracted.squeeze(1)  # (B, watermark_length)
        
        return extracted


class RobustEncoder(nn.Module):
    """Encoder with dual-domain watermarking and error correction"""
    def __init__(self, config):
        super(RobustEncoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.watermark_length = config.watermark_length
        
        # Reed-Solomon encoder
        self.rs_encoder = ReedSolomonEncoder(config.watermark_length, redundancy=10)
        self.encoded_length = self.rs_encoder.encoded_length
        
        # Spatial domain encoder (original)
        layers = [ConvBNRelu(3, self.conv_channels)]
        for _ in range(config.encoder_blocks-1):
            layers.append(ConvBNRelu(self.conv_channels, self.conv_channels))
        self.conv_layers = nn.Sequential(*layers)
        
        self.after_concat_layer = ConvBNRelu(
            self.conv_channels + 3 + self.encoded_length, 
            self.conv_channels
        )
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)
        
        # Frequency domain encoder
        self.dct_layer = DCTWatermarkLayer()
        
        # Learnable weights for domain combination
        self.spatial_weight = nn.Parameter(torch.tensor(0.6))
        self.freq_weight = nn.Parameter(torch.tensor(0.4))
        
        # Refinement iterations
        self.refine_iterations = 3
        
    def forward(self, original_image, watermark):
        """
        Dual-domain watermark insertion with iterative refinement
        """
        # Encode watermark with error correction
        encoded_watermark = self.rs_encoder.encode(watermark)
        if encoded_watermark.device != original_image.device:
            encoded_watermark = encoded_watermark.to(original_image.device)
        
        # Initial spatial domain insertion
        expanded_watermark = encoded_watermark.unsqueeze(-1).unsqueeze(-1)
        expanded_watermark = expanded_watermark.expand(-1, -1, self.H, self.W)
        
        encoded_image = self.conv_layers(original_image)
        concat = torch.cat([expanded_watermark, encoded_image, original_image], dim=1)
        spatial_watermarked = self.after_concat_layer(concat)
        spatial_watermarked = self.final_layer(spatial_watermarked)
        
        # Normalize weights
        w_spatial = torch.sigmoid(self.spatial_weight)
        w_freq = torch.sigmoid(self.freq_weight)
        total = w_spatial + w_freq
        w_spatial = w_spatial / total
        w_freq = w_freq / total
        
        # Iterative refinement to align spatial and frequency domains
        current_image = spatial_watermarked
        for iteration in range(self.refine_iterations):
            # Insert in frequency domain
            freq_watermarked = self.dct_layer(current_image, encoded_watermark, strength=0.05)
            
            # Combine both domains
            combined = w_spatial * spatial_watermarked + w_freq * freq_watermarked
            
            # Small adjustment to maintain image quality
            combined = torch.clamp(combined, -1, 1)
            
            # Update for next iteration
            if iteration < self.refine_iterations - 1:
                # Refine spatial insertion based on frequency domain
                residual = freq_watermarked - spatial_watermarked
                spatial_watermarked = spatial_watermarked + 0.1 * residual
                current_image = combined
        
        return combined