"""Device selection and placement: env override, CPU fallback, adapters put models on the selected device.

All models are tiny random-init configs; nothing is downloaded. The "meta" device stands in for cuda/mps so the
placement path is exercised on any machine.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from fairlens import hardware, models

TINY_BERT = dict(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                 max_position_embeddings=32)


class FakeModule(torch.nn.Module):
    """Module whose .to() refuses some devices and whose forward can fail once off-CPU."""

    def __init__(self, refuse: set[str] = frozenset(), fail_off_cpu: bool = False):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.refuse, self.fail_off_cpu, self.current, self.calls = refuse, fail_off_cpu, "cpu", 0

    def to(self, dev):
        if str(dev) in self.refuse:
            raise RuntimeError(f"{dev} unavailable")
        self.current = str(dev)
        return self

    def run(self):
        self.calls += 1
        if self.fail_off_cpu and self.current != "cpu":
            raise NotImplementedError("op not implemented for this device")
        return self.current


def _tiny_bert_cls():
    from transformers import BertConfig, BertForSequenceClassification

    return BertForSequenceClassification(BertConfig(**TINY_BERT, num_labels=2))


@pytest.fixture(autouse=True)
def _skip_numeric_selfcheck(monkeypatch):
    """These tests fake device availability; the numerical self-check (tested in test_reproducibility) is skipped."""
    monkeypatch.setenv("FAIRLENS_SKIP_SELFCHECK", "1")


@pytest.fixture
def fresh_manager(monkeypatch):
    monkeypatch.setattr(models, "_MANAGER", models.ModelManager(max_resident_mb=100_000))


@pytest.fixture
def on_meta(monkeypatch):
    monkeypatch.setattr(models, "selected_device", lambda: "meta")


def _no_accelerators(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_mps_available", lambda _t: False)


def test_env_override_selects_cpu_even_with_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("FAIRLENS_DEVICE", "CPU")
    assert hardware.device() == "cpu"


def test_env_override_selects_available_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("FAIRLENS_DEVICE", "cuda")
    assert hardware.device() == "cuda"


def test_preference_order_without_override(monkeypatch):
    monkeypatch.delenv("FAIRLENS_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware, "_mps_available", lambda _t: True)
    assert hardware.device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert hardware.device() == "mps"
    _no_accelerators(monkeypatch)
    assert hardware.device() == "cpu"


@pytest.mark.parametrize("forced", ["cuda", "mps"])
def test_forced_unavailable_device_falls_back_to_cpu(monkeypatch, forced):
    _no_accelerators(monkeypatch)
    monkeypatch.setenv("FAIRLENS_DEVICE", forced)
    with pytest.warns(UserWarning, match="falling back to CPU"):
        assert hardware.device() == "cpu"


def test_place_falls_back_to_cpu_when_move_fails():
    m = FakeModule(refuse={"cuda"})
    with pytest.warns(UserWarning, match="using CPU"):
        assert models.place(m, "cuda").current == "cpu"


def test_place_keeps_mps_unsafe_models_on_cpu(monkeypatch):
    monkeypatch.setattr(models, "selected_device", lambda: "mps")
    assert models.place(FakeModule(), allow_mps=False).current == "cpu"
    assert models.place(FakeModule()).current == "mps"


def test_runtime_failure_off_cpu_retries_on_cpu():
    m = FakeModule(fail_off_cpu=True)
    m.current = "mps"
    m.w = torch.nn.Parameter(torch.zeros(1, device="meta"))
    with pytest.warns(UserWarning, match="retrying on CPU"):
        assert models.run_with_cpu_fallback(m, m.run) == "cpu"
    assert m.calls == 2


def test_runtime_failure_on_cpu_is_raised():
    m = FakeModule()
    with pytest.raises(ZeroDivisionError):
        models.run_with_cpu_fallback(m, lambda: 1 / 0)


def test_to_model_device_moves_dict_inputs():
    lin = torch.nn.Linear(2, 2).to("meta")
    out = models.to_model_device({"x": torch.zeros(2), "n": 3}, lin)
    assert out["x"].device.type == "meta" and out["n"] == 3


def test_hf_detector_places_model_on_selected_device(monkeypatch, fresh_manager, on_meta):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from fairlens.detection.hf_classifier import HFClassifierDetector, ZeroShotNLIDetector

    monkeypatch.setattr(AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: _tiny_bert_cls())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    _, model = HFClassifierDetector("t", "T", "tiny/model", biased_label_ids=[1])._load()
    assert models.model_device(model).type == "meta"
    _, nli = ZeroShotNLIDetector("z", "Z", "tiny/nli", {"biased": "b"})._load()
    assert models.model_device(nli).type == "meta"


def test_hf_detector_scores_on_cpu_with_moved_inputs(monkeypatch, fresh_manager):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from fairlens.detection.hf_classifier import HFClassifierDetector

    monkeypatch.setattr(models, "selected_device", lambda: "cpu")
    monkeypatch.setattr(AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: _tiny_bert_cls())
    fake_tok = lambda texts, **k: {"input_ids": torch.ones(len(texts), 4, dtype=torch.long),  # noqa: E731
                                   "attention_mask": torch.ones(len(texts), 4, dtype=torch.long)}
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: fake_tok)
    scores = HFClassifierDetector("t", "T", "tiny/model", biased_label_ids=[1])._scores(["a", "b"])
    assert len(scores) == 2 and all(0 <= s <= 1 for s in scores)


def test_rewriter_places_model_on_selected_device(monkeypatch, fresh_manager, on_meta):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, T5ForConditionalGeneration

    from fairlens.mitigation.llm_adapters import Seq2SeqRewriter

    cfg = T5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2)
    monkeypatch.setattr(AutoModelForSeq2SeqLM, "from_pretrained", lambda *a, **k: T5ForConditionalGeneration(cfg))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    _, model = Seq2SeqRewriter("g", "G", "tiny/t5")._load()
    assert models.model_device(model).type == "meta"


def test_preservation_models_placed_on_selected_device(monkeypatch, fresh_manager, on_meta):
    import sentence_transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from fairlens.evaluation.preservation import Entailment, SemanticSimilarity

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *a, **k: torch.nn.Linear(2, 2))
    monkeypatch.setattr(AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: _tiny_bert_cls())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    assert models.model_device(SemanticSimilarity("tiny/emb")._load()).type == "meta"
    _, nli = Entailment("tiny/nli")._load()
    assert models.model_device(nli).type == "meta"


def test_clip_placed_on_selected_device(monkeypatch, fresh_manager, on_meta):
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor

    from fairlens.multimodal.caption_evaluation import ClipAlignment

    cfg = CLIPConfig(text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                      num_attention_heads=2, max_position_embeddings=16),
                     vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                        num_attention_heads=2, image_size=32, patch_size=16),
                     projection_dim=8)
    monkeypatch.setattr(CLIPModel, "from_pretrained", lambda *a, **k: CLIPModel(cfg))
    monkeypatch.setattr(CLIPProcessor, "from_pretrained", lambda *a, **k: object())
    _, model = ClipAlignment("tiny/clip")._load()
    assert models.model_device(model).type == "meta"


def _train_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_detectors.py"
    spec = importlib.util.spec_from_file_location("train_detectors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_freeze_embeddings_freezes_only_the_embedding_layer():
    td = _train_script()
    model = _tiny_bert_cls()
    n_emb = sum(p.numel() for p in model.base_model.embeddings.parameters())
    assert td.freeze_embeddings(model) == n_emb
    assert not any(p.requires_grad for p in model.base_model.embeddings.parameters())
    assert all(p.requires_grad for p in model.base_model.encoder.parameters())
    assert td.HPARAMS["deberta"]["freeze_embeddings"] and not td.HPARAMS["bert"]["freeze_embeddings"]
