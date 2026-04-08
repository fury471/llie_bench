# Benchmark Protocol

## Channel Convention
- sRGB methods must be evaluated on the Y channel (luminance)
- RAW methods must be evaluated on the RAW channel
- PSNR and SSIM are computed on Y channel only
- Do not mix sRGB methods with RAW datasets

## Fairness Rules
- All methods use the same random seed (default: 42)
- All methods are evaluated on the same test split
- No test-time augmentation unless explicitly stated
- No fine-tuning on test data

## Input Convention
- All images normalized to [0, 1] range
- All images in CHW format (channels, height, width)
- Batch size must be consistent across methods