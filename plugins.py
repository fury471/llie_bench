# Central plugin registry — import all plugins here
# Every new plugin must be added to this file

# Methods
import methods.zerodce
import methods.clahe
import methods.retinexnet

# Datasets
import datasets_loaders.lolv1
import datasets_loaders.lolv2

# Metrics
import metrics.psnr
import metrics.ssim
import metrics.niqe