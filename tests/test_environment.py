import os

from fairlens import config, hardware


def test_config_loads_and_declares_all_slots():
    cfg = config.load_config()
    assert {"rule_based", "bert", "deberta", "bart_mnli"} <= set(cfg["detectors"])
    assert {"rule_based", "flan_t5", "qwen"} <= set(cfg["mitigation"]["rewriters"])
    assert {"blip", "florence"} <= set(cfg["captioning"]["captioners"])
    assert 0 < cfg["gate"]["min_semantic_similarity"] <= 1


def test_hf_cache_is_configured():
    assert os.environ.get("HF_HOME")


def test_hardware_detection_and_profile(monkeypatch):
    info = hardware.detect(check_internet=False)
    assert info.ram_total_gb > 0 and info.cpu_threads >= 1
    assert hardware.select_profile(info) in {"gpu", "standard", "lite"}
    monkeypatch.setenv("FAIRLENS_PROFILE", "lite")
    assert hardware.select_profile(info) == "lite"
