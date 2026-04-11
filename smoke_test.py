import torch
import plugins
from core.registry import METHOD_REGISTRY, DATASET_REGISTRY, METRIC_REGISTRY, lookup
from core.config import load_config

SEPARATOR = "-" * 50

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def ok(msg):
    print(f"  ✓ {msg}")

def test_registry():
    section("Registry")
    assert "zerodce" in METHOD_REGISTRY, "ZeroDCE not registered"
    ok("zerodce registered")
    assert "clahe" in METHOD_REGISTRY, "CLAHE not registered"
    ok("clahe registered")
    assert "retinexnet" in METHOD_REGISTRY, "RetinexNet not registered"
    ok("retinexnet registered")
    assert "lolv1" in DATASET_REGISTRY, "LOLv1 not registered"
    ok("lolv1 registered")
    assert "lolv2" in DATASET_REGISTRY, "LOLv2 not registered"
    ok("lolv2 registered")
    assert "psnr" in METRIC_REGISTRY, "PSNR not registered"
    ok("psnr registered")
    assert "ssim" in METRIC_REGISTRY, "SSIM not registered"
    ok("ssim registered")
    assert "niqe" in METRIC_REGISTRY, "NIQE not registered"
    ok("niqe registered")

def test_config():
    section("Config Merging")
    config = load_config("configs/experiments/full_bench_lolv1.yaml")
    assert "method" in config, "method key missing"
    ok(f"method:    {config['method']}")
    assert "dataset" in config, "dataset key missing"
    ok(f"dataset:   {config['dataset']}")
    assert "metrics" in config, "metrics key missing"
    ok(f"metrics:   {config['metrics']}")
    assert "lr" in config, "lr key missing — method config not merged"
    ok(f"lr:        {config['lr']}")
    assert "data_root" in config, "data_root key missing — dataset config not merged"
    ok(f"data_root: {config['data_root']}")

def test_method(method_name, input_shape, expected_output_shape):
    section(f"Method: {method_name}")
    model = lookup(METHOD_REGISTRY, method_name)()
    fake_batch = [torch.rand(*input_shape).clamp(0, 1)]
    output = model.enhance(fake_batch)
    assert output.shape == torch.Size(expected_output_shape), \
        f"Expected {expected_output_shape}, got {output.shape}"
    ok(f"output shape: {output.shape}")

def test_metrics():
    section("Metrics")
    pred = torch.rand(3, 64, 64)
    gt = torch.rand(3, 64, 64)
    for metric_name in ["psnr", "ssim"]:
        metric = lookup(METRIC_REGISTRY, metric_name)()
        pred = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        metric.compute(pred, gt)
        result = metric.aggregate()
        ok(f"{metric_name}: {result:.4f}")

    # NIQE needs larger images
    niqe = lookup(METRIC_REGISTRY, "niqe")()
    pred_large = torch.rand(3, 128, 128)
    niqe.compute(pred_large, None)
    ok(f"niqe: {niqe.aggregate():.4f}")

def test_dataset(dataset_name, data_root, split="test"):
    section(f"Dataset: {dataset_name}")
    dataset = lookup(DATASET_REGISTRY, dataset_name)(data_root, split=split)
    sample = dataset[0]
    assert len(sample) == 2, "Expected [low, high] pair"
    ok(f"size:       {len(dataset)}")
    low, high = sample
    assert low.shape == high.shape, "Low and high shapes must match"
    ok(f"low shape:  {low.shape}")
    ok(f"high shape: {high.shape}")
    assert low.max() <= 1.0 and low.min() >= 0.0, "Images must be normalized to [0, 1]"
    ok(f"normalized: [0, 1]")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  LLIE BENCH — SMOKE TEST")
    print("=" * 50)

    test_registry()
    test_config()
    test_method("zerodce", (1, 3, 64, 64), (1, 3, 64, 64))
    test_method("clahe",   (1, 3, 64, 64), (1, 3, 64, 64))
    test_method("retinexnet", (1, 3, 64, 64), (1, 3, 64, 64))
    test_metrics()
    test_dataset("lolv1", "data/LOLdataset", split="test")
    test_dataset("lolv2", "data/LOL-v2/LOL-v2", split="Test")

    print(f"\n{'=' * 50}")
    print("  ALL TESTS PASSED")
    print("=" * 50 + "\n")