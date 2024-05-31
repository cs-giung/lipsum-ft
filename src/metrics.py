"""Evaluation metrics for classification tasks."""
import jax
import jax.numpy as jnp


def evaluate_acc(confidences, true_labels, reduction="mean"):
    """Classification accuracy."""
    pred_labels = jnp.argmax(confidences, axis=1)
    raw_results = jnp.equal(pred_labels, true_labels)
    if reduction == "none":
        return raw_results
    if reduction == "mean":
        return jnp.mean(raw_results)
    if reduction == "sum":
        return jnp.sum(raw_results)
    raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def evaluate_nll(
        confidences, true_labels, reduction="mean", log_input=True, eps=1e-8):
    """Negative log likelihood."""
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    true_target = jax.nn.one_hot(
        true_labels, num_classes=log_confidences.shape[1])
    raw_results = -jnp.sum(
        jnp.where(true_target, true_target * log_confidences, 0.0), axis=-1)
    if reduction == "none":
        return raw_results
    if reduction == "mean":
        return jnp.mean(raw_results)
    if reduction == "sum":
        return jnp.sum(raw_results)
    raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')
