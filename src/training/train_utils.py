from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F


DEFAULT_RECALL_KS = (1, 5, 10)
GRAD_CLIP_NORM = 1.0


def contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Standard CLIP-style contrastive loss using exact image-text pairs.
    """
    scale = logit_scale.exp().clamp(max=100)
    logits = image_embeds @ text_embeds.T * scale
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


def finding_aware_contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    findings: Sequence[str],
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-positive contrastive loss where samples with the same finding
    are treated as positives for one another.
    """
    scale = logit_scale.exp().clamp(max=100)
    logits = image_embeds @ text_embeds.T * scale

    findings = list(findings)
    device = logits.device

    # Map finding strings to integer ids, then build mask via broadcasting
    finding_to_idx = {f: i for i, f in enumerate(set(findings))}
    finding_ids = torch.tensor([finding_to_idx[f] for f in findings], device=device)
    positive_mask = (finding_ids.unsqueeze(0) == finding_ids.unsqueeze(1)).float()

    log_probs_i2t = F.log_softmax(logits, dim=1)
    positives_per_row_i2t = positive_mask.sum(dim=1).clamp(min=1.0)
    loss_i2t = -(positive_mask * log_probs_i2t).sum(dim=1) / positives_per_row_i2t
    loss_i2t = loss_i2t.mean()

    log_probs_t2i = F.log_softmax(logits.T, dim=1)
    positive_mask_t = positive_mask.T
    positives_per_row_t2i = positive_mask_t.sum(dim=1).clamp(min=1.0)
    loss_t2i = -(positive_mask_t * log_probs_t2i).sum(dim=1) / positives_per_row_t2i
    loss_t2i = loss_t2i.mean()

    return (loss_i2t + loss_t2i) / 2


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    log_every: int = 50,
    use_finding_aware_loss: bool = False,
) -> Dict[str, float]:
    """
    Train the model for one epoch and return summary metrics.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        findings = batch.get("finding")

        optimizer.zero_grad()

        image_embeds, text_embeds = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=images,
        )

        if use_finding_aware_loss:
            if findings is None:
                raise ValueError("Batch is missing 'finding' but finding-aware loss is enabled.")

            loss = finding_aware_contrastive_loss(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                findings=findings,
                logit_scale=model.logit_scale,
            )
        else:
            loss = contrastive_loss(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                logit_scale=model.logit_scale,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        num_batches += 1

        if step % log_every == 0:
            print(f"step {step}/{len(loader)} train_loss={loss.item():.4f}")

    average_loss = running_loss / max(num_batches, 1)
    return {"train_loss": average_loss}


@torch.no_grad()
def _encode_dataset(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, Any]:
    """
    Encode the full dataset into image and text embeddings and collect metadata.
    """
    model.eval()

    all_image_embeds = []
    all_text_embeds = []

    metadata = {
        "image_path": [],
        "report_text": [],
        "study_id": [],
        "finding": [],
        "negation_text": [],
        "omitted_text": [],
        "location": [],
    }

    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_embeds, text_embeds = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=images,
        )

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())

        batch_size = images.size(0)
        for key in metadata:
            metadata[key].extend(batch.get(key, [""] * batch_size))

    return {
        "image_embeds": torch.cat(all_image_embeds, dim=0),
        "text_embeds": torch.cat(all_text_embeds, dim=0),
        "metadata": metadata,
    }


def _compute_exact_recall(
    similarity: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int],
) -> Dict[str, float]:
    """
    Compute exact Recall@K for a similarity matrix.
    """
    metrics: Dict[str, float] = {}

    for k in ks:
        actual_k = min(k, similarity.size(1))
        topk_indices = similarity.topk(k=actual_k, dim=1).indices
        hit_rate = (topk_indices == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"r{k}_exact"] = hit_rate

    return metrics


def _compute_finding_recall(
    similarity: torch.Tensor,
    findings: Sequence[str],
    ks: Sequence[int],
) -> Dict[str, float]:
    """
    Compute Recall@K where any sample with the same finding counts as correct.
    """
    metrics: Dict[str, float] = {}

    for k in ks:
        actual_k = min(k, similarity.size(1))
        topk_indices = similarity.topk(k=actual_k, dim=1).indices

        hits = []
        for query_idx in range(similarity.size(0)):
            true_finding = findings[query_idx]
            retrieved_indices = topk_indices[query_idx].tolist()
            hit = any(findings[idx] == true_finding for idx in retrieved_indices)
            hits.append(hit)

        metrics[f"r{k}_finding"] = sum(hits) / max(len(hits), 1)

    return metrics


def _build_i2t_analysis(
    similarity: torch.Tensor,
    metadata: Dict[str, List[Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Build per-query retrieval analysis for image-to-text retrieval.
    """
    findings = metadata["finding"]
    report_texts = metadata["report_text"]
    study_ids = metadata["study_id"]
    negation_texts = metadata["negation_text"]
    omitted_texts = metadata["omitted_text"]
    locations = metadata["location"]
    image_paths = metadata["image_path"]

    topk_scores, topk_indices = similarity.topk(k=min(top_k, similarity.size(1)), dim=1)

    analysis = []

    for query_idx in range(similarity.size(0)):
        retrieved_items = []

        for rank, (score, retrieved_idx) in enumerate(
            zip(topk_scores[query_idx].tolist(), topk_indices[query_idx].tolist()),
            start=1,
        ):
            retrieved_items.append(
                {
                    "rank": rank,
                    "retrieved_index": retrieved_idx,
                    "score": score,
                    "report_text": report_texts[retrieved_idx],
                    "study_id": study_ids[retrieved_idx],
                    "finding": findings[retrieved_idx],
                    "negation_text": negation_texts[retrieved_idx],
                    "omitted_text": omitted_texts[retrieved_idx],
                    "location": locations[retrieved_idx],
                    "image_path": image_paths[retrieved_idx],
                }
            )

        top1_idx = topk_indices[query_idx, 0].item()
        retrieved_topk = topk_indices[query_idx].tolist()

        analysis.append(
            {
                "query_index": query_idx,
                "query_image_path": image_paths[query_idx],
                "query_study_id": study_ids[query_idx],
                "true_index": query_idx,
                "true_report_text": report_texts[query_idx],
                "true_finding": findings[query_idx],
                "true_negation_text": negation_texts[query_idx],
                "true_omitted_text": omitted_texts[query_idx],
                "true_location": locations[query_idx],
                "topk": retrieved_items,
                "is_top1_exact_correct": top1_idx == query_idx,
                "is_topk_exact_correct": query_idx in retrieved_topk,
                "is_top1_finding_correct": findings[top1_idx] == findings[query_idx],
                "is_topk_finding_correct": any(
                    findings[idx] == findings[query_idx] for idx in retrieved_topk
                ),
            }
        )

    return analysis


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    top_k: int = 5,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
) -> Dict[str, Any]:
    """
    Evaluate retrieval performance for both exact-match and finding-level retrieval.
    """
    encoded = _encode_dataset(model=model, loader=loader, device=device)

    image_embeds = encoded["image_embeds"]
    text_embeds = encoded["text_embeds"]
    metadata = encoded["metadata"]
    findings = metadata["finding"]

    similarity_i2t = image_embeds @ text_embeds.T
    similarity_t2i = similarity_i2t.T
    targets = torch.arange(similarity_i2t.size(0))

    i2t_exact = _compute_exact_recall(similarity_i2t, targets, recall_ks)
    t2i_exact = _compute_exact_recall(similarity_t2i, targets, recall_ks)

    i2t_finding = _compute_finding_recall(similarity_i2t, findings, recall_ks)
    t2i_finding = _compute_finding_recall(similarity_t2i, findings, recall_ks)

    metrics = {}
    for key, value in i2t_exact.items():
        metrics[f"i2t_{key}"] = value
    for key, value in t2i_exact.items():
        metrics[f"t2i_{key}"] = value
    for key, value in i2t_finding.items():
        metrics[f"i2t_{key}"] = value
    for key, value in t2i_finding.items():
        metrics[f"t2i_{key}"] = value

    i2t_analysis = _build_i2t_analysis(
        similarity=similarity_i2t,
        metadata=metadata,
        top_k=top_k,
    )

    return {
        "metrics": metrics,
        "i2t_analysis": i2t_analysis,
    }