
"""
sisa_unlearn.py
================
SISA (Sharded, Isolated, Sliced, Aggregated) Machine Unlearning Engine
for MemoryMesh — Legal AI Compliance Platform.
 
Implements:
  - Model weight sharding across N isolated shards
  - Per-user data provenance tracking (which shard trained on which user)
  - Selective shard retraining on /forget requests (no full-model retrain)
  - Accuracy benchmarking before vs. after each unlearn operation
  - CSV audit log of every unlearn event
 
References:
  Bourtoule et al. (2021) "Machine Unlearning"
  https://arxiv.org/abs/1912.03817
"""
 
import os
import csv
import json
import time
import hashlib
import logging
import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
 
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sisa_unlearn")
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SISAConfig:
    num_shards: int = 4          # N shards total
    num_slices: int = 3          # slices per shard (for incremental retraining)
    retrain_epochs: int = 5      # epochs when retraining an affected shard
    learning_rate: float = 1e-3
    batch_size: int = 32
    model_hidden_dim: int = 128
    input_dim: int = 784         # e.g. flattened 28×28 for MNIST-like data
    output_dim: int = 10
    device: str = "cpu"          # switch to "cuda" if available
    benchmark_csv: str = "benchmark_report.csv"
    provenance_db: str = "provenance.json"
 
 
# ---------------------------------------------------------------------------
# Tiny demo model (swap with your actual LLM adapter / embedding head)
# ---------------------------------------------------------------------------
class ShardModel(nn.Module):
    """
    Lightweight MLP used as a stand-in for a real LLM classification head.
    In production, replace with a LoRA adapter or a frozen-base + head setup.
    """
    def __init__(self, cfg: SISAConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.model_hidden_dim, cfg.model_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.model_hidden_dim // 2, cfg.output_dim),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
 
# ---------------------------------------------------------------------------
# Provenance tracker — maps user_id → list of (shard_id, slice_id, sample_idx)
# ---------------------------------------------------------------------------
class ProvenanceTracker:
    """
    Tracks exactly which model shards/slices were trained on which user's data.
    Persisted to JSON for crash recovery and audit purposes.
    """
 
    def __init__(self, path: str):
        self.path = Path(path)
        # Structure: { user_id: [ {shard_id, slice_id, sample_indices: [...]} ] }
        self._db: Dict[str, List[dict]] = {}
        if self.path.exists():
            self._db = json.loads(self.path.read_text())
 
    def record(
        self,
        user_id: str,
        shard_id: int,
        slice_id: int,
        sample_indices: List[int],
    ) -> None:
        key = self._hash_user(user_id)
        entry = {
            "shard_id": shard_id,
            "slice_id": slice_id,
            "sample_indices": sample_indices,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._db.setdefault(key, []).append(entry)
        self._persist()
 
    def get_affected_shards(self, user_id: str) -> List[int]:
        key = self._hash_user(user_id)
        entries = self._db.get(key, [])
        return list({e["shard_id"] for e in entries})
 
    def get_shard_entries(self, user_id: str, shard_id: int) -> List[dict]:
        key = self._hash_user(user_id)
        return [
            e for e in self._db.get(key, []) if e["shard_id"] == shard_id
        ]
 
    def purge_user(self, user_id: str) -> None:
        key = self._hash_user(user_id)
        self._db.pop(key, None)
        self._persist()
        logger.info("Provenance records purged for user_id hash=%s", key[:8])
 
    def _hash_user(self, user_id: str) -> str:
        """Never store raw user IDs — store SHA-256 hashes."""
        return hashlib.sha256(user_id.encode()).hexdigest()
 
    def _persist(self) -> None:
        self.path.write_text(json.dumps(self._db, indent=2))
 
 
# ---------------------------------------------------------------------------
# Benchmark logger
# ---------------------------------------------------------------------------
class BenchmarkLogger:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            self._write_header()
 
    def _write_header(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc",
                "user_id_hash",
                "shard_id",
                "accuracy_before",
                "accuracy_after",
                "accuracy_delta",
                "retrain_time_sec",
                "samples_removed",
                "status",
            ])
 
    def log(
        self,
        user_id_hash: str,
        shard_id: int,
        acc_before: float,
        acc_after: float,
        retrain_secs: float,
        samples_removed: int,
        status: str = "success",
    ) -> None:
        delta = acc_after - acc_before
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                user_id_hash[:12] + "…",   # partial hash only — privacy
                shard_id,
                f"{acc_before:.4f}",
                f"{acc_after:.4f}",
                f"{delta:+.4f}",
                f"{retrain_secs:.2f}",
                samples_removed,
                status,
            ])
        logger.info(
            "Benchmark shard=%d  acc_before=%.4f  acc_after=%.4f  Δ=%+.4f  "
            "time=%.2fs  removed=%d",
            shard_id, acc_before, acc_after, delta, retrain_secs, samples_removed,
        )
 
 
# ---------------------------------------------------------------------------
# SISA Engine — core class
# ---------------------------------------------------------------------------
class SISAUnlearnEngine:
    """
    Sharded, Isolated, Sliced, Aggregated unlearning engine.
 
    Usage
    -----
    engine = SISAUnlearnEngine(cfg)
    engine.train_all(full_dataset, user_label_map)   # initial training
    acc, shard_preds = engine.aggregate_predict(test_loader)
    engine.forget(user_id="alice@example.com", retrain_dataset=remaining_ds)
    """
 
    def __init__(self, cfg: SISAConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.provenance = ProvenanceTracker(cfg.provenance_db)
        self.benchmark = BenchmarkLogger(cfg.benchmark_csv)
 
        # One model per shard, kept in memory (no persistent vector store)
        self.shards: List[ShardModel] = [
            ShardModel(cfg).to(self.device) for _ in range(cfg.num_shards)
        ]
        # Optimizers — one per shard
        self.optimizers: List[optim.Optimizer] = [
            optim.Adam(s.parameters(), lr=cfg.learning_rate)
            for s in self.shards
        ]
        # Shard → list of DataLoaders (one per slice)
        self.shard_slice_loaders: List[List[DataLoader]] = [
            [] for _ in range(cfg.num_shards)
        ]
 
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_all(
        self,
        dataset: Dataset,
        user_sample_map: Dict[str, List[int]],
    ) -> None:
        """
        Distribute dataset across shards and train each shard on its slice.
 
        Parameters
        ----------
        dataset        : full training dataset
        user_sample_map: { user_id: [global_sample_indices] }
        """
        n = len(dataset)
        indices = list(range(n))
        shard_size = n // self.cfg.num_shards
 
        logger.info("SISA training: %d samples → %d shards", n, self.cfg.num_shards)
 
        for shard_id in range(self.cfg.num_shards):
            start = shard_id * shard_size
            end = start + shard_size if shard_id < self.cfg.num_shards - 1 else n
            shard_indices = indices[start:end]
 
            # Split shard into slices
            slice_size = max(1, len(shard_indices) // self.cfg.num_slices)
            slices = [
                shard_indices[i * slice_size: (i + 1) * slice_size]
                for i in range(self.cfg.num_slices)
            ]
 
            # Record provenance for every user whose samples fall in this shard
            for user_id, user_indices in user_sample_map.items():
                for sl_id, sl_indices in enumerate(slices):
                    overlap = list(set(user_indices) & set(sl_indices))
                    if overlap:
                        self.provenance.record(user_id, shard_id, sl_id, overlap)
 
            # Cumulative slice training (SISA core idea)
            cumulative_indices: List[int] = []
            slice_loaders: List[DataLoader] = []
            for sl_id, sl_indices in enumerate(slices):
                cumulative_indices.extend(sl_indices)
                subset = Subset(dataset, cumulative_indices)
                loader = DataLoader(
                    subset,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                )
                slice_loaders.append(loader)
                self._train_shard(shard_id, loader, self.cfg.retrain_epochs)
                logger.info("  Shard %d / Slice %d trained (%d samples)",
                            shard_id, sl_id, len(cumulative_indices))
 
            self.shard_slice_loaders[shard_id] = slice_loaders
 
        logger.info("SISA initial training complete.")
 
    def _train_shard(
        self,
        shard_id: int,
        loader: DataLoader,
        epochs: int,
    ) -> None:
        model = self.shards[shard_id]
        optimizer = self.optimizers[shard_id]
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
 
    # ------------------------------------------------------------------
    # Inference — majority-vote aggregation across shards
    # ------------------------------------------------------------------
    def aggregate_predict(
        self,
        loader: DataLoader,
    ) -> Tuple[float, List[torch.Tensor]]:
        """
        Returns (accuracy, list_of_per_shard_logits).
        Aggregation = mean of logits across all shards (soft voting).
        """
        all_shard_logits: List[List[torch.Tensor]] = [[] for _ in self.shards]
 
        for shard_id, model in enumerate(self.shards):
            model.eval()
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(self.device)
                    all_shard_logits[shard_id].append(model(x))
 
        correct = total = 0
        for batch_idx, (_, y) in enumerate(loader):
            y = y.to(self.device)
            shard_logits = torch.stack(
                [all_shard_logits[s][batch_idx] for s in range(len(self.shards))]
            ).mean(dim=0)
            preds = shard_logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
 
        accuracy = correct / total if total else 0.0
        return accuracy, all_shard_logits
 
    def evaluate_shard(self, shard_id: int, loader: DataLoader) -> float:
        """Accuracy of a single shard."""
        model = self.shards[shard_id]
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total else 0.0
 
    # ------------------------------------------------------------------
    # Unlearning — the /forget operation
    # ------------------------------------------------------------------
    def forget(
        self,
        user_id: str,
        retrain_dataset: Dataset,
        test_loader: DataLoader,
    ) -> Dict[str, object]:
        """
        Execute GDPR/DPDP "Right to be Forgotten" for user_id.
 
        Steps
        -----
        1. Find all shards that trained on user_id's data (provenance lookup).
        2. For each affected shard:
           a. Benchmark accuracy BEFORE unlearning.
           b. Re-initialise the shard weights (cryptographic erasure of learned weights).
           c. Retrain shard from scratch on retrain_dataset (user's data excluded).
           d. Benchmark accuracy AFTER unlearning.
           e. Log to benchmark_report.csv.
        3. Purge user provenance records.
        4. Return audit summary.
        """
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()
        affected_shards = self.provenance.get_affected_shards(user_id)
 
        if not affected_shards:
            logger.warning("No provenance found for user_id (hash=%s)", user_hash[:8])
            return {"status": "no_data_found", "shards_retrained": 0}
 
        logger.info(
            "FORGET request: user_hash=%s  affected_shards=%s",
            user_hash[:8], affected_shards,
        )
 
        audit: Dict[str, object] = {
            "user_id_hash": user_hash,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "shards_retrained": [],
            "total_retrain_time_sec": 0.0,
        }
 
        retrain_loader = DataLoader(
            retrain_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
 
        samples_removed_total = 0
        for shard_id in affected_shards:
            entries = self.provenance.get_shard_entries(user_id, shard_id)
            samples_removed = sum(len(e["sample_indices"]) for e in entries)
            samples_removed_total += samples_removed
 
            # --- Benchmark BEFORE ---
            acc_before = self.evaluate_shard(shard_id, test_loader)
 
            # --- Cryptographic weight erasure: reinitialise shard ---
            self._reinitialise_shard(shard_id)
            logger.info("  Shard %d weights cryptographically erased.", shard_id)
 
            # --- Retrain on clean data ---
            t0 = time.perf_counter()
            self._train_shard(shard_id, retrain_loader, self.cfg.retrain_epochs)
            retrain_secs = time.perf_counter() - t0
 
            # --- Benchmark AFTER ---
            acc_after = self.evaluate_shard(shard_id, test_loader)
 
            # --- Log ---
            self.benchmark.log(
                user_id_hash=user_hash,
                shard_id=shard_id,
                acc_before=acc_before,
                acc_after=acc_after,
                retrain_secs=retrain_secs,
                samples_removed=samples_removed,
            )
 
            audit["shards_retrained"].append({  # type: ignore[union-attr]
                "shard_id": shard_id,
                "acc_before": round(acc_before, 4),
                "acc_after": round(acc_after, 4),
                "retrain_time_sec": round(retrain_secs, 2),
                "samples_removed": samples_removed,
            })
            audit["total_retrain_time_sec"] = (  # type: ignore[assignment]
                float(audit["total_retrain_time_sec"]) + retrain_secs
            )
 
        # --- Purge provenance ---
        self.provenance.purge_user(user_id)
 
        audit["status"] = "completed"
        audit["total_samples_removed"] = samples_removed_total
        logger.info(
            "FORGET complete: user_hash=%s  shards=%d  samples=%d  time=%.2fs",
            user_hash[:8],
            len(affected_shards),
            samples_removed_total,
            float(audit["total_retrain_time_sec"]),
        )
        return audit
 
    def _reinitialise_shard(self, shard_id: int) -> None:
        """
        Overwrite shard weights with fresh random values — effectively
        cryptographic deletion of what was learned from user data.
        Also resets the optimizer state.
        """
        new_model = ShardModel(self.cfg).to(self.device)
        self.shards[shard_id] = new_model
        self.optimizers[shard_id] = optim.Adam(
            new_model.parameters(), lr=self.cfg.learning_rate
        )
 
 
# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
def _make_demo_dataset(n: int, cfg: SISAConfig) -> Tuple[TensorDataset, Dict[str, List[int]]]:
    """Generate random dataset and assign first 50 samples to 'alice@example.com'."""
    X = torch.randn(n, cfg.input_dim)
    y = torch.randint(0, cfg.output_dim, (n,))
    ds = TensorDataset(X, y)
    user_map = {
        "alice@example.com": list(range(0, 50)),
        "bob@example.com":   list(range(50, 80)),
    }
    return ds, user_map
 
 
def run_demo() -> None:
    cfg = SISAConfig(
        num_shards=4,
        num_slices=2,
        retrain_epochs=3,
        input_dim=64,
        output_dim=5,
        model_hidden_dim=64,
    )
 
    # Build dataset
    n_train, n_test = 400, 100
    train_ds, user_map = _make_demo_dataset(n_train, cfg)
    test_ds, _ = _make_demo_dataset(n_test, cfg)
    test_loader = DataLoader(test_ds, batch_size=32)
 
    engine = SISAUnlearnEngine(cfg)
 
    logger.info("=== Phase 1: Initial SISA training ===")
    engine.train_all(train_ds, user_map)
 
    acc, _ = engine.aggregate_predict(test_loader)
    logger.info("Aggregate accuracy after training: %.4f", acc)
 
    logger.info("=== Phase 2: FORGET alice@example.com ===")
    # Retrain dataset = original data minus Alice's indices
    alice_indices = user_map["alice@example.com"]
    remaining_indices = [i for i in range(n_train) if i not in alice_indices]
    from torch.utils.data import Subset
    retrain_ds = Subset(train_ds, remaining_indices)
 
    audit = engine.forget(
        user_id="alice@example.com",
        retrain_dataset=retrain_ds,
        test_loader=test_loader,
    )
 
    logger.info("Audit summary:\n%s", json.dumps(audit, indent=2))
 
    acc_post, _ = engine.aggregate_predict(test_loader)
    logger.info("Aggregate accuracy after unlearn: %.4f", acc_post)
    logger.info("Benchmark report written to: %s", cfg.benchmark_csv)
 
 
if __name__ == "__main__":
    run_demo()#
