import torch
from methods.zerodce import ZeroDCE
from datasets.lolv1 import LOLv1
from metrics.psnr import PSNRMetric
from metrics.ssim import SSIMMetric
from core.registry import METHOD_REGISTRY, DATASET_REGISTRY, METRIC_REGISTRY, lookup

def test_registry():
    print("Testing registry...")
    assert "zerodce" in METHOD_REGISTRY, "ZeroDCE not registered"
    assert "lolv1" in DATASET_REGISTRY, "LOLv1 not registered"
    assert "psnr" in METRIC_REGISTRY, "PSNR not registered"
    assert "ssim" in METRIC_REGISTRY, "SSIM not registered"
    print("Registry OK")

def test_model():
    print("Testing ZeroDCE model...")
    model = ZeroDCE()
    fake_batch = [torch.randn(1, 3, 64, 64)]
    output = model.forward(fake_batch)
    assert output.shape == (1, 24, 64, 64), f"Unexpected output shape: {output.shape}"
    print(f"Model output shape: {output.shape} OK")

def test_metrics():
    print("Testing metrics...")
    pred = torch.rand(3, 64, 64)
    gt = torch.rand(3, 64, 64)
    
    psnr = PSNRMetric()
    psnr.compute(pred, gt)
    print(f"PSNR: {psnr.aggregate():.2f} OK")
    
    ssim = SSIMMetric()
    ssim.compute(pred, gt)
    print(f"SSIM: {ssim.aggregate():.4f} OK")

def test_dataset():
    print("Testing LOLv1 dataset...")
    dataset = LOLv1("data/LOLdataset", split="test")
    print(f"Dataset size: {len(dataset)}")
    low, high = dataset[0]
    print(f"Low image shape: {low.shape} OK")
    print(f"High image shape: {high.shape} OK")

if __name__ == "__main__":
    test_registry()
    test_model()
    test_metrics()
    test_dataset()
    print("\nAll tests passed!")