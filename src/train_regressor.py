import os
from typing import Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from prettytable import PrettyTable

from dora import get_xp, hydra_main

from .sae import SparseAutoencoder
from .data import initialize_dataloaders
from .utils import (
    get_logger,
    configure_runtime,
    should_disable_tqdm,
    metrics_from_counts,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _flatten_tokens_with_labels(batch,device):
    embeddings = batch["embeddings"].to(device, non_blocking=True)
    mask = batch["attention_mask"].to(device, non_blocking=True).bool()
    
    if not mask.any():
        return None
    if "ner_tags" not in batch:
        return embeddings[mask], None
    
    labels_raw = batch["ner_tags"].to(device, non_blocking=True)
    labels_flat = labels_raw[mask]
    entities = (labels_flat > 0).to(torch.bool)  # treat any non-zero tag as entity

    tokens = embeddings[mask]
    return tokens, entities

class SparseAETrainer:
    def __init__(self, cfg, train_dl, eval_dl, dev_dl, logger, xp, device) -> None:
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.xp = xp

        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.dev_dl = dev_dl

        self.grad_clip = float(cfg.sae.grad_clip)
        self.log_interval = int(cfg.train.log_interval)
        self.checkpoint_path = cfg.train.checkpoint
        self.disable_progress = should_disable_tqdm() or cfg.train.grid_mode

        self.model: Optional[SparseAutoencoder] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.global_step = 0

    def _init_model(self, tokens: torch.Tensor):
        if self.model is not None:
            return
        sae_cfg = self.cfg.sae
        d_model = tokens.size(-1)
        self.model = SparseAutoencoder(
            d_model=d_model,
            expansion=sae_cfg.expansion,
            alpha=sae_cfg.alpha,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(sae_cfg.lr),
            weight_decay=float(sae_cfg.weight_decay),
        )
        if not self.cfg.train.grid_mode:
            self.logger.info(
                "Initialized SAE: d_model=%d d_hidden=%d expansion=%.2f alpha=%.6f",
                self.model.d_model,
                self.model.d_hidden,
                self.model.expansion,
                self.model.alpha,
            )

    def _metrics_table(self, metrics: Dict[str, float]) -> PrettyTable:
        table = PrettyTable()
        table.field_names = list(metrics.keys())
        row = [f"{v:.6f}" if isinstance(v, float) else v for v in metrics.values()]
        table.add_row(row)
        return table

    def _train_epoch(self, epoch_idx: int) -> Dict[str, float]:
        if self.model is not None:
            self.model.train()
        totals = {"loss": 0.0, "mse": 0.0, "l1": 0.0}
        token_count = 0

        iterator = tqdm(self.train_dl, desc=f"SAE Train {epoch_idx + 1}", disable=self.disable_progress)
        for batch in iterator:
            tokens, _ = _flatten_tokens_with_labels(batch, self.device)
            if tokens is None:
                continue
            self._init_model(tokens)
            self.model.train()

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(tokens)
            loss = output["loss"]
            loss.backward()

            if self.grad_clip > 0.0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.model.normalize_dictionary()

            num_tokens = tokens.size(0)
            token_count += num_tokens
            totals["loss"] += loss.item() * num_tokens
            totals["mse"] += float(output["mse"].detach()) * num_tokens
            totals["l1"] += float(output["l1"].detach())

            self.global_step += 1

        if token_count == 0:
            raise RuntimeError("No tokens were processed in this epoch; check that the dataset has ner_tags and non-empty batches.")

        return totals["loss"] / token_count, totals["mse"] / token_count, totals["l1"] / token_count
    
    def _save_checkpoint(self):
        if self.model is None:
            raise RuntimeError("Cannot save checkpoint without a model.")
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "meta": {
                "d_model": self.model.d_model,
                "d_hidden": self.model.d_hidden,
                "expansion": self.model.expansion,
                "alpha": self.model.alpha,
            },
        }
        torch.save(state, self.checkpoint_path, _use_new_zipfile_serialization=False)
        if not self.cfg.train.grid_mode:
            self.logger.info("Saved SAE checkpoint to %s", os.path.abspath(self.checkpoint_path))

    def _load_checkpoint(self):
        path = self.checkpoint_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        state = torch.load(path, map_location=self.device)
        meta = state.get("meta", {})
        d_model = int(meta.get("d_model"))
        expansion = float(meta.get("expansion", self.cfg.sae.expansion))
        alpha = float(meta.get("alpha", self.cfg.sae.alpha))
        self.model = SparseAutoencoder(d_model=d_model, expansion=expansion, alpha=alpha).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.sae.lr),
            weight_decay=float(self.cfg.sae.weight_decay),
        )
        self.model.load_state_dict(state["model"], strict=True)
        if "optimizer" in state and state["optimizer"] is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        if not self.cfg.train.grid_mode:
            self.logger.info(
                "Loaded SAE checkpoint from %s (d_model=%d d_hidden=%d expansion=%.2f alpha=%.6f)",
                path,
                self.model.d_model,
                self.model.d_hidden,
            self.model.expansion,
            self.model.alpha,
        )

    def _select_entity_dimension(self, loader) -> Optional[int]:
        """
        Pure PyTorch implementation of a sparse (L1) logistic regression probe
        to identify which SAE dictionary dimension best predicts entityness.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.eval()

        codes_list = []
        labels_list = []

        # ---- Collect representations ----
        with torch.no_grad():
            iterator = tqdm(loader, desc="Collecting probe data", disable=self.disable_progress)
            for batch in iterator:
                flattened = _flatten_tokens_with_labels(batch, self.device)
                if flattened is None:
                    continue
                tokens, labels = flattened  # labels: boolean mask
                out = self.model(tokens)
                codes = out["codes"]  # [num_tokens, d_hidden]

                codes_list.append(codes)
                labels_list.append(labels.float())

        if len(codes_list) == 0:
            self.logger.warning("No data for probe.")
            return None

        X = torch.cat(codes_list, dim=0).to(self.device)      # [N, d_hidden]
        y = torch.cat(labels_list, dim=0).to(self.device)      # [N]

        if y.sum() == 0 or y.sum() == len(y):
            self.logger.warning("Probe failed: dataset lacks entity/non-entity contrast.")
            return None

        d_hidden = X.size(-1)

        # ---- Define logistic regression model ----
        w = torch.zeros(d_hidden, device=self.device, requires_grad=True)
        b = torch.zeros(1, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([w, b], lr=1e-2)

        lambda_l1 = 1.0  # sparsity strength; tune if needed
        batch_size = 4096
        num_steps = 200

        # ---- Train the sparse probe ----
        for _ in range(num_steps):
            idx = torch.randint(0, X.size(0), (batch_size,), device=self.device)
            xb = X[idx]
            yb = y[idx]

            logits = xb @ w + b
            preds = torch.sigmoid(logits)

            bce = torch.nn.functional.binary_cross_entropy(preds, yb)
            l1_penalty = lambda_l1 * w.abs().sum()

            loss = bce + l1_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return w.detach(), b.detach()

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate entity prediction using the full sparse logistic probe
        (w, b) learned by _select_entity_dimension.
        """
        # 1. Train sparse probe on eval_dl
        result = self._select_entity_dimension(self.eval_dl)
        if result is None:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        w, b = result  # each from _select_entity_dimension
        w = w.to(self.device)
        b = b.to(self.device)

        # threshold for sigmoid output
        thresh = float(self.cfg.entity_eval.threshold)

        tp = fp = fn = 0
        self.model.eval()

        with torch.no_grad():
            iterator = tqdm(self.dev_dl, desc="Entity probe eval", disable=self.disable_progress)
            for batch in iterator:
                flattened = _flatten_tokens_with_labels(batch, self.device)
                if flattened is None:
                    continue
                tokens, labels = flattened  # labels: bool mask

                codes = self.model(tokens)["codes"]  # [num_tokens, d_hidden]
                logits = codes @ w + b              # [num_tokens]
                probs = torch.sigmoid(logits)
                preds = probs > thresh

                gold = labels

                tp += int((preds & gold).sum().item())
                fp += int((preds & ~gold).sum().item())
                fn += int((~preds & gold).sum().item())

        f1, precision, recall = metrics_from_counts(tp, fp, fn)

        return f1, precision, recall

    def train(self):
        best_f1 = 0.0
        for epoch in range(int(self.cfg.train.epochs)):
            loss, mse, l1 = self._train_epoch(epoch)
            if not self.cfg.train.grid_mode:
                self.logger.info(
                    "Epoch %d/%d train metrics:\n%s",
                    epoch + 1,
                    self.cfg.train.epochs,
                    self._metrics_table({"loss": loss, "mse": mse, "l1": l1}),
                )
            f1, precision, recall = self.evaluate()
            if not self.cfg.train.grid_mode:
                self.logger.info("Epoch %d metrics:\n%s", epoch + 1, self._metrics_table({"f1": f1, "precision": precision, "recall": recall}))
            if f1 > best_f1:
                best_f1 = f1
                self.xp.link.push_metrics({"best_f1": best_f1, "best_epoch": epoch + 1})
                self.logger.info("Epoch %d metrics:\n%s", epoch + 1, self._metrics_table({"f1": f1, "precision": precision, "recall": recall}))
            self._save_checkpoint()


@hydra_main(config_path="conf", config_name="default", version_base="1.1")
def main(cfg):
    logger = get_logger("train_sae.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    if not cfg.train.grid_mode:
        logger.info(repr(cfg))
        logger.info(f"Work dir: {os.getcwd()}")

    configure_runtime(cfg)
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available() and not cfg.train.grid_mode:
        logger.warning("CUDA requested but unavailable, using CPU.")
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    cfg.runtime.device = device.type

    train_dl, eval_dl, dev_dl = initialize_dataloaders(cfg, log=not cfg.train.grid_mode)
    trainer = SparseAETrainer(cfg, train_dl, eval_dl, dev_dl, logger, xp, device)

    if cfg.train.eval_only:
        trainer._load_checkpoint()
        metrics = trainer.evaluate(eval_dl, desc="SAE Eval")
        logger.info("Evaluation metrics:\n%s", trainer._metrics_table(metrics))
        f1, precision, recall = trainer.evaluate()
        logger.info("Entity metrics:\n%s", trainer._metrics_table({"f1": f1, "precision": precision, "recall": recall}))
    else:
        trainer.train()


if __name__ == "__main__":
    main()
