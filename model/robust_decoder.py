import torch
import torch.nn as nn
import torch.nn.functional as F
from model.conv_bn_relu import ConvBNRelu
from model.robust_encoder import ReedSolomonEncoder, DCTWatermarkLayer


class RobustDecoder(nn.Module):
    """Decoder with dual-domain extraction and error correction"""
    def __init__(self, config):
        super(RobustDecoder, self).__init__()
        self.channels = config.decoder_channels
        self.watermark_length = config.watermark_length
        
        # Reed-Solomon decoder
        self.rs_encoder = ReedSolomonEncoder(config.watermark_length, redundancy=10)
        self.encoded_length = self.rs_encoder.encoded_length
        
        # Spatial domain decoder (original approach)
        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))
        layers.append(ConvBNRelu(self.channels, self.encoded_length))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.spatial_layers = nn.Sequential(*layers)
        
        self.spatial_linear = nn.Linear(self.encoded_length, self.encoded_length)
        
        # Frequency domain decoder
        self.dct_layer = DCTWatermarkLayer()
        self.freq_linear = nn.Linear(self.encoded_length, self.encoded_length)
        
        # Cross-validation layers
        self.fusion_layer = nn.Linear(self.encoded_length * 2, self.encoded_length)
        self.confidence_layer = nn.Linear(self.encoded_length * 2, 2)  # confidence scores for each domain
        
    def forward(self, watermarked_image):
        """
        Extract watermark from both domains and validate
        """
        # Extract from spatial domain
        spatial_features = self.spatial_layers(watermarked_image)
        spatial_features.squeeze_(3).squeeze_(2)
        spatial_watermark = self.spatial_linear(spatial_features)
        
        # Extract from frequency domain
        freq_watermark = self.dct_layer.extract(watermarked_image, self.encoded_length)
        freq_watermark = self.freq_linear(freq_watermark)
        
        # Cross-validation: compute confidence scores
        combined_features = torch.cat([spatial_watermark, freq_watermark], dim=1)
        confidence = torch.softmax(self.confidence_layer(combined_features), dim=1)
        
        # Weighted fusion based on confidence
        spatial_conf = confidence[:, 0:1]
        freq_conf = confidence[:, 1:2]
        
        fused_watermark = (spatial_conf * spatial_watermark + 
                          freq_conf * freq_watermark)
        
        # Additional fusion with learned weights
        fused_watermark = self.fusion_layer(combined_features)
        
        # Decode with error correction
        decoded_watermark = self.rs_encoder.decode(fused_watermark)
        
        # Return only the message part (without redundancy)
        return decoded_watermark[:, :self.watermark_length]
    
    def extract_with_validation(self, watermarked_image):
        """
        Extract watermark and return validation metrics
        Returns:
            watermark: decoded watermark
            spatial_raw: raw spatial domain extraction
            freq_raw: raw frequency domain extraction
            agreement: agreement score between domains
        """
        # Extract from both domains
        spatial_features = self.spatial_layers(watermarked_image)
        spatial_features.squeeze_(3).squeeze_(2)
        spatial_watermark = self.spatial_linear(spatial_features)
        
        freq_watermark = self.dct_layer.extract(watermarked_image, self.encoded_length)
        freq_watermark = self.freq_linear(freq_watermark)
        
        # Compute agreement score
        spatial_bits = torch.round(torch.sigmoid(spatial_watermark))
        freq_bits = torch.round(torch.sigmoid(freq_watermark))
        agreement = (spatial_bits == freq_bits).float().mean(dim=1)
        
        # Fuse and decode
        combined_features = torch.cat([spatial_watermark, freq_watermark], dim=1)
        fused_watermark = self.fusion_layer(combined_features)
        decoded_watermark = self.rs_encoder.decode(fused_watermark)
        
        return {
            'watermark': decoded_watermark[:, :self.watermark_length],
            'spatial_raw': spatial_watermark,
            'freq_raw': freq_watermark,
            'agreement': agreement,
            'fused': fused_watermark
        }