import os
from typing import Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer
import numpy as np
import copy

from dora import get_xp, hydra_main

from .bucket_tree import BucketTree
from .data import initialize_dataloaders
from .utils import get_logger, configure_runtime, should_disable_tqdm, metrics_from_counts

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _prepare_batch(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    embeddings = batch["embeddings"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    if "ner_tags" in batch:
        return embeddings, attention_mask, input_ids, batch["ner_tags"].to(device, non_blocking=True)
    return embeddings, attention_mask, input_ids, None


class BucketTrainer:
    def __init__(self, cfg, train_dl, eval_dl, logger, xp, device) -> None:
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.xp = xp
        self.train_dl = train_dl
        self.eval_dl = eval_dl

        self.grad_clip = float(cfg.bucket_model.grad_clip)
        self.checkpoint_path = cfg.train.checkpoint
        self.disable_progress = should_disable_tqdm()

        self.loss_weights = {
            "entropy_weight": float(cfg.bucket_model.entropy_weight),
            "balance_weight": float(cfg.bucket_model.balance_weight),
            "prototype_pull_weight": float(cfg.bucket_model.prototype_pull_weight),
            "prototype_repulsion_weight": float(cfg.bucket_model.prototype_repulsion_weight),
        }

        d_model = self._infer_d_model()
        bm_cfg = copy.deepcopy(self.cfg.bucket_model)
        bm_cfg.d_model = d_model
        self.tree = BucketTree(
            bucket_cfg=bm_cfg,
            loss_weights=self.loss_weights,
            device=self.device,
            sbert_name=self.cfg.bucket_model.sbert_model,
        )

    def _iter_nodes(self):
        stack = [self.tree.root]
        while stack:
            node = stack.pop()
            yield node
            for child in node.children:
                stack.append(child)

    def _save_checkpoint(self):
        state = {"nodes": {}, "meta": {"tree_depth": self.tree.depth}}
        for node in self._iter_nodes():
            key = "root" if node.path == () else "_".join(map(str, node.path))
            state["nodes"][key] = {
                "model": node.model.state_dict(),
                "optimizer": node.optimizer.state_dict(),
                "path": node.path,
            }
        torch.save(state, self.checkpoint_path, _use_new_zipfile_serialization=False)
        self.logger.info("Saved bucket tree checkpoint to %s", os.path.abspath(self.checkpoint_path))

    def _load_checkpoint(self):
        path = self.checkpoint_path
        state = torch.load(path, map_location=self.device)
        nodes_state = state.get("nodes", {})
        for node in self._iter_nodes():
            key = "root" if node.path == () else "_".join(map(str, node.path))
            if key not in nodes_state:
                continue
            node.model.load_state_dict(nodes_state[key]["model"], strict=False)
            try:
                node.optimizer.load_state_dict(nodes_state[key]["optimizer"])
            except Exception:
                pass
        self.logger.info("Loaded bucket tree checkpoint from %s", path)

    def _infer_d_model(self) -> int:
        """
        Determine embedding dimension from dataset or a single batch.
        """
        ds = getattr(self.train_dl, "dataset", None)
        sample = None
        if ds is not None and len(ds) > 0:
            sample = ds[0]
        if sample is None:
            try:
                sample_batch = next(iter(self.train_dl))
                sample = sample_batch
            except Exception as err:
                raise RuntimeError("Unable to infer d_model from training data.") from err

        emb = sample["embeddings"]
        if isinstance(emb, torch.Tensor):
            return int(emb.shape[-1])
        if isinstance(emb, np.ndarray):
            return int(emb.shape[-1])
        try:
            return int(len(emb[-1]))
        except Exception as err:
            raise RuntimeError("Unable to infer d_model from dataset sample.") from err

    def _metrics_table(self, metrics: Dict[str, float]) -> PrettyTable:
        table = PrettyTable()
        table.field_names = list(metrics.keys())
        row = [f"{v:.6f}" if isinstance(v, float) else v for v in metrics.values()]
        table.add_row(row)
        return table

    def _run_batch(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, input_ids: torch.Tensor, *, train: bool) -> Dict[str, torch.Tensor]:
        # Zero all node grads
        for node in self._iter_nodes():
            node.optimizer.zero_grad(set_to_none=True)

        leaves = self.tree.train_forward(embeddings, attention_mask, input_ids)
        if not leaves:
            return {}, torch.tensor(0.0, device=self.device)
        loss = sum(entry["losses"]["loss"] for entry in leaves)

        if train:
            loss.backward()
            if self.grad_clip > 0.0:
                for node in self._iter_nodes():
                    clip_grad_norm_(node.model.parameters(), self.grad_clip)
            for node in self._iter_nodes():
                node.optimizer.step()
        # Aggregate metrics across leaves
        metrics = {}
        num = len(leaves)
        for entry in leaves:
            for k, v in entry["losses"].items():
                val = v if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device, dtype=torch.float32)
                metrics[k] = metrics.get(k, 0.0) + float(val.detach())
        for k in metrics:
            metrics[k] /= num
        return metrics, loss

    def _train_epoch(self, epoch_idx: int) -> Dict[str, float]:
        totals = {
            "loss": 0.0,
            "recon": 0.0,
            "cos": 0.0,
            "entropy": 0.0,
            "balance": 0.0,
            "pull": 0.0,
            "repulsion": 0.0,
        }
        example_count = 0

        iterator = tqdm(self.train_dl, desc=f"Training {epoch_idx + 1}", disable=self.disable_progress)
        for batch in iterator:
            embeddings, attention_mask, input_ids, _ = _prepare_batch(batch, self.device)
            metrics, loss = self._run_batch(embeddings, attention_mask, input_ids, train=True)

            batch_size = embeddings.size(0)
            example_count += batch_size
            totals["loss"] += float(loss.detach()) * batch_size
            totals["recon"] += float(metrics["recon"]) * batch_size
            totals["cos"] += float(metrics["cos"]) * batch_size
            totals["entropy"] += float(metrics["entropy"]) * batch_size
            totals["balance"] += float(metrics["balance"]) * batch_size
            totals["pull"] += float(metrics["pull"]) * batch_size
            totals["repulsion"] += float(metrics["repulsion"]) * batch_size
        if example_count == 0:
            raise RuntimeError("No training examples were processed.")

        return {k: v / example_count for k, v in totals.items()}

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        counts = {}
        iterator = tqdm(self.eval_dl, desc="Bucket Eval (entity)", disable=self.disable_progress)
        for batch in iterator:
            embeddings, attention_mask, input_ids, labels = _prepare_batch(batch, self.device)
            if labels is None:
                continue
            gold = (labels > 0) & attention_mask.bool()
            results = self.tree.eval_forward(embeddings, attention_mask, input_ids)
            for path, gates, mask in results:
                mask_bool = mask.bool()
                for k in range(gates.size(-1)):
                    pred = gates[:, :, k] & mask_bool
                    key = f"{'root' if not path else '_'.join(map(str, path))}:{k}"
                    tp, fp, fn = counts.get(key, (0.0, 0.0, 0.0))
                    tp += (pred & gold).sum().item()
                    fp += (pred & ~gold).sum().item()
                    fn += ((~pred) & gold).sum().item()
                    counts[key] = (tp, fp, fn)

        metrics = []
        for key, (tp, fp, fn) in counts.items():
            f1, p, r = metrics_from_counts(tp, fp, fn)
            metrics.append((key, f1, p, r, tp, fp, fn))
        if not metrics:
            return -1, 0.0, 0.0, 0.0, "No eval data"

        metrics_sorted = sorted(metrics, key=lambda x: x[1], reverse=True)
        best_key, best_f1, best_p, best_r, _, _, _ = metrics_sorted[0]

        table = PrettyTable()
        table.field_names = ["bucket", "f1", "precision", "recall", "tp", "fp", "fn"]
        for key, f1, p, r, tp, fp, fn in metrics_sorted:
            table.add_row([key, f"{f1:.4f}", f"{p:.4f}", f"{r:.4f}", int(tp), int(fp), int(fn)])

        return best_key, best_f1, best_p, best_r, table.get_string()

    def evaluate(self, header) -> None:
        best_idx, best_f1, best_p, best_r, table_str = self._evaluate()
        self.logger.info(header+"\n"+table_str)
        self.logger.info(
            "Best bucket=%s f1=%.4f precision=%.4f recall=%.4f",
            best_idx,
            best_f1,
            best_p,
            best_r,
        )

    def train_stage(self, stage: int, epochs: int):
        for epoch in tqdm(range(epochs), desc=f"Stage {stage} Training", disable=self.disable_progress):
            train_metrics = self._train_epoch(epoch)
            self.logger.info(
                "Epoch %d/%d train:\n%s",
                epoch + 1,
                epochs,
                self._metrics_table(
                    {
                        "loss": train_metrics["loss"],
                        "recon": train_metrics["recon"],
                        "cos": train_metrics["cos"],
                        "entropy": train_metrics["entropy"],
                        "balance": train_metrics["balance"],
                        "pull": train_metrics["pull"],
                        "repulsion": train_metrics["repulsion"],
                    }
                ),
            )
            if self.eval_dl is not None and self.cfg.train.eval_epoch:
                self.evaluate(f"Stage {stage} Epoch {epoch + 1} Evaluation:")
            self._save_checkpoint()

    def train(self):
        for stage_idx, epochs in enumerate(self.cfg.train.epochs):
            self.train_stage(stage_idx + 1, epochs)
            self.evaluate(f"Stage {stage_idx + 1}:")


@hydra_main(config_path="conf", config_name="bucket", version_base="1.1")
def main(cfg):
    logger = get_logger("train_bucket.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    configure_runtime(cfg)
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, using CPU.")
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    cfg.runtime.device = device.type

    train_dl, eval_dl, _ = initialize_dataloaders(cfg, logger)
    trainer = BucketTrainer(cfg, train_dl, eval_dl, logger, xp, device)

    if cfg.train.eval_only:
        trainer._load_checkpoint()
        best_idx, best_f1, best_p, best_r, table_str = trainer.evaluate()
        logger.info("Eval-only table:\n%s", table_str)
        logger.info(
            "Eval-only: best_bucket=%s f1=%.4f precision=%.4f recall=%.4f",
            best_idx,
            best_f1,
            best_p, 
            best_r,
        )
    else:
        trainer.train()


if __name__ == "__main__":
    main()
