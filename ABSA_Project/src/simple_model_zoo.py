"""
simple_model_zoo.py
===================
Classical / simple models for UIT-ViSFD ABSA, implemented as a multi-head
classifier: one 4-class classifier per aspect (10 heads), all sharing the same
TF-IDF representation.

Why multi-head?
---------------
The task label is a set of up to 10 ``{ASPECT#Polarity}`` pairs.  A multi-head
output preserves the project's fixed aspect order and the 4-class label order
(None / Negative / Neutral / Positive), exactly like the PhoBERT checkpoint.
It also lets us report ACD (mention vs None) and ASC (polarity) separately.

Models
------
majority, stratified, multinomial_nb, complement_nb, bernoulli_nb,
logistic_regression, linear_svc, ridge, sgd_log, decision_tree,
random_forest, extra_trees, mlp, knn

``sgd_log`` and ``mlp`` are trained one epoch at a time with ``partial_fit`` so
that train/dev loss and accuracy can be logged after every epoch.  The other
models are closed-form, batch or ensemble learners; per-epoch curves are not
defined for them and this is stated in RESULTS.txt.

Author: Nhom 2 - CT255
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit, softmax
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from experiment_metrics import LABELS, normalize_probabilities


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "majority": {
        "kind": "baseline",
        "needs_features": False,
        "epochs": False,
        "description": "Per-aspect majority class (most frequent label in Train).",
    },
    "stratified": {
        "kind": "baseline",
        "needs_features": False,
        "epochs": False,
        "description": "Per-aspect random draw from the Train class prior (seeded).",
    },
    "multinomial_nb": {
        "kind": "linear",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + Multinomial Naive Bayes, one head per aspect.",
    },
    "complement_nb": {
        "kind": "linear",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + Complement Naive Bayes, one head per aspect.",
    },
    "bernoulli_nb": {
        "kind": "linear",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + Bernoulli Naive Bayes (binarised features).",
    },
    "logistic_regression": {
        "kind": "linear",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + multinomial Logistic Regression (L2), one head per aspect.",
    },
    "linear_svc": {
        "kind": "linear",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + Linear Support Vector Classifier; decision scores -> softmax.",
    },
    "ridge": {
        "kind": "linear",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + Ridge Classifier; decision scores -> softmax.",
    },
    "sgd_log": {
        "kind": "linear",
        "needs_features": True,
        "epochs": True,
        "description": "TF-IDF + SGD logistic loss, partial_fit per epoch (loss/acc curve).",
    },
    "decision_tree": {
        "kind": "tree",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + CART Decision Tree, one head per aspect.",
    },
    "random_forest": {
        "kind": "ensemble",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + Random Forest, one head per aspect.",
    },
    "extra_trees": {
        "kind": "ensemble",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + Extremely Randomised Trees, one head per aspect.",
    },
    "mlp": {
        "kind": "neural",
        "needs_features": True,
        "epochs": True,
        "description": "TF-IDF + sklearn MLP, partial_fit per epoch (loss/acc curve).",
    },
    "knn": {
        "kind": "instance",
        "needs_features": True,
        "epochs": False,
        "description": "TF-IDF + cosine k-NN, one head per aspect (slower).",
    },
}

DEFAULT_MODELS = [
    "majority",
    "stratified",
    "multinomial_nb",
    "complement_nb",
    "logistic_regression",
    "linear_svc",
    "ridge",
    "sgd_log",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "mlp",
]

ALL_MODELS = list(MODEL_REGISTRY.keys())
EPOCH_MODELS = {"sgd_log", "mlp"}
NO_FEATURE_MODELS = {"majority", "stratified"}


def describe_models() -> str:
    lines = []
    for name, info in MODEL_REGISTRY.items():
        flag = "epoch-curve" if info["epochs"] else "single-fit"
        lines.append(f"  {name:<22} {info['kind']:<10} {flag:<12} {info['description']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def build_vectorizer(args) -> Any:
    """Word TF-IDF, optionally concatenated with character n-grams."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=args.sublinear_tf,
        lowercase=args.lowercase,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    if args.char_ngram_max and args.char_ngram_max >= 2:
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, args.char_ngram_max),
            max_features=max(1000, args.max_features // 2),
            min_df=args.min_df,
            max_df=args.max_df,
            sublinear_tf=args.sublinear_tf,
            lowercase=args.lowercase,
            dtype=np.float32,
        )
        return FeatureUnion([("word", word), ("char", char)])
    return word


def vectorizer_description(args) -> Dict[str, Any]:
    return {
        "type": "word TF-IDF" + (" + char_wb TF-IDF" if args.char_ngram_max and args.char_ngram_max >= 2 else ""),
        "word_ngram_range": [args.ngram_min, args.ngram_max],
        "char_ngram_range": [2, args.char_ngram_max] if args.char_ngram_max and args.char_ngram_max >= 2 else None,
        "max_features": args.max_features,
        "char_max_features": max(1000, args.max_features // 2) if args.char_ngram_max and args.char_ngram_max >= 2 else None,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "sublinear_tf": bool(args.sublinear_tf),
        "lowercase": bool(args.lowercase),
        "token_pattern": r"(?u)\b\w+\b",
    }


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------
def _class_weight(args, name: str):
    if args.class_weight == "none":
        return None
    if name in {"random_forest", "extra_trees"} and args.class_weight == "balanced_subsample":
        return "balanced_subsample"
    return "balanced"


def balanced_weight_dict(y: np.ndarray) -> Dict[int, float]:
    """Class weights for estimators whose partial_fit rejects the string 'balanced'."""
    classes, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
    n = int(counts.sum())
    k = max(1, len(classes))
    return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}


def build_estimator(name: str, args, seed: int):
    cw = _class_weight(args, name)
    if name == "logistic_regression":
        return LogisticRegression(
            C=args.C,
            class_weight=cw,
            max_iter=args.max_iter,
            solver=args.solver,
            random_state=seed,
        )
    if name == "linear_svc":
        return LinearSVC(
            C=args.C,
            class_weight=cw,
            max_iter=args.max_iter,
            dual="auto",
            random_state=seed,
        )
    if name == "ridge":
        return RidgeClassifier(
            alpha=args.alpha,
            class_weight=cw,
            random_state=seed,
        )
    if name == "multinomial_nb":
        return MultinomialNB(alpha=args.alpha)
    if name == "complement_nb":
        return ComplementNB(alpha=args.alpha)
    if name == "bernoulli_nb":
        return BernoulliNB(alpha=args.alpha, binarize=0.0)
    if name == "sgd_log":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=args.sgd_alpha,
            max_iter=1,
            tol=None,
            class_weight=None,  # set per head with explicit weights (partial_fit limitation)
            learning_rate="optimal",
            random_state=seed,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=cw,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=cw,
            n_jobs=args.jobs,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=cw,
            n_jobs=args.jobs,
            random_state=seed,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(args.hidden_layers),
            activation="relu",
            solver="adam",
            alpha=args.mlp_alpha,
            batch_size=args.batch_size,
            learning_rate_init=args.learning_rate,
            max_iter=1,
            shuffle=True,
            random_state=seed,
        )
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=args.knn_neighbors,
            metric="cosine",
            weights="distance",
            algorithm="brute",
            n_jobs=args.jobs,
        )
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Heads and multi-head wrapper
# ---------------------------------------------------------------------------
@dataclass
class Head:
    name: str
    estimator: Any = None
    classes: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    const_class: int = 0
    const_prior: np.ndarray = field(default_factory=lambda: np.full(4, 0.25))
    n_iter: Optional[int] = None
    train_seconds: float = 0.0

    @property
    def is_constant(self) -> bool:
        return self.estimator is None


def make_constant_head(name: str, y: np.ndarray) -> Head:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=4).astype(float)
    prior = counts / counts.sum() if counts.sum() > 0 else np.full(4, 0.25)
    return Head(
        name=name,
        estimator=None,
        classes=np.array([int(np.argmax(prior))], dtype=int),
        const_class=int(np.argmax(prior)),
        const_prior=prior,
    )


def head_probabilities(head: Head, X) -> np.ndarray:
    """Return an (n, 4) probability matrix mapped to the global label order."""
    n = X.shape[0]
    if head.is_constant:
        return np.tile(head.const_prior, (n, 1))

    est = head.estimator
    if hasattr(est, "predict_proba"):
        raw = np.asarray(est.predict_proba(X), dtype=float)
        classes = np.asarray(est.classes_, dtype=int)
    elif hasattr(est, "decision_function"):
        decision = np.asarray(est.decision_function(X), dtype=float)
        classes = np.asarray(est.classes_, dtype=int)
        if decision.ndim == 1:
            p1 = expit(decision)
            raw = np.column_stack([1.0 - p1, p1])
        else:
            raw = softmax(decision, axis=1)
    else:
        pred = np.asarray(est.predict(X), dtype=int)
        classes = np.asarray(est.classes_, dtype=int)
        raw = np.zeros((n, len(classes)), dtype=float)
        for i, c in enumerate(classes):
            raw[pred == c, i] = 1.0

    full = np.zeros((n, 4), dtype=float)
    for i, c in enumerate(classes):
        if 0 <= int(c) < 4:
            full[:, int(c)] = raw[:, i]
    # If a class was absent from training, it gets zero mass; renormalise.
    return normalize_probabilities(full)


class MultiHeadModel:
    """A shared vectorizer plus 10 independent 4-class heads."""

    def __init__(self, name: str, heads: List[Head], vectorizer: Any, n_features: int):
        self.name = name
        self.heads = heads
        self.vectorizer = vectorizer
        self.n_features = int(n_features)
        self._rng = np.random.default_rng(0)

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0]
        proba = np.zeros((n, len(self.heads), 4), dtype=float)
        for a, head in enumerate(self.heads):
            proba[:, a, :] = head_probabilities(head, X)
        return proba

    def predict(self, X) -> np.ndarray:
        if self.name == "stratified":
            proba = self.predict_proba(X)
            n, n_heads, _ = proba.shape
            out = np.zeros((n, n_heads), dtype=int)
            for a in range(n_heads):
                out[:, a] = self._rng.choice(4, size=n, p=self.heads[a].const_prior)
            return out
        return self.predict_proba(X).argmax(axis=2)

    def head_n_iter(self) -> List[Optional[int]]:
        return [h.n_iter for h in self.heads]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _fit_one_head(name: str, X_train, y: np.ndarray, args, seed: int) -> Head:
    classes = np.unique(y)
    if len(classes) <= 1:
        return make_constant_head(name, y)
    est = build_estimator(name, args, seed)
    start = time.perf_counter()
    est.fit(X_train, y)
    elapsed = time.perf_counter() - start
    n_iter = getattr(est, "n_iter_", None)
    if isinstance(n_iter, np.ndarray):
        n_iter = int(np.max(n_iter)) if n_iter.size else None
    elif n_iter is not None:
        n_iter = int(n_iter)
    return Head(
        name=name,
        estimator=est,
        classes=np.asarray(est.classes_, dtype=int),
        n_iter=n_iter,
        train_seconds=elapsed,
    )


def _epoch_eval(
    model: MultiHeadModel, X, Y: np.ndarray
) -> Tuple[float, float, float]:
    """Aggregate log-loss, accuracy and macro-F1 over the 10 heads."""
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=2)
    losses, accs = [], []
    for a in range(Y.shape[1]):
        p = normalize_probabilities(proba[:, a, :])
        if len(np.unique(Y[:, a])) >= 2:
            losses.append(log_loss(Y[:, a], p, labels=LABELS))
        accs.append(accuracy_score(Y[:, a], pred[:, a]))
    loss = float(np.mean(losses)) if losses else 0.0
    acc = float(np.mean(accs)) if accs else 0.0
    macro_f1 = float(f1_score(Y.ravel(), pred.ravel(), average="macro", zero_division=0))
    return loss, acc, macro_f1


def fit_simple_model(
    name: str,
    X_train,
    Y_train: np.ndarray,
    X_dev,
    Y_dev: Optional[np.ndarray],
    args,
    seed: int,
) -> Tuple[MultiHeadModel, List[Dict[str, Any]]]:
    """
    Train one multi-head model.

    Returns the fitted model and an epoch log (empty for single-fit models).
    """
    n_heads = Y_train.shape[1]
    epoch_log: List[Dict[str, Any]] = []

    if name in NO_FEATURE_MODELS:
        heads = [make_constant_head(name, Y_train[:, a]) for a in range(n_heads)]
        model = MultiHeadModel(name, heads, None, 0)
        model._rng = np.random.default_rng(seed)
        return model, epoch_log

    heads: List[Head] = []
    if name in EPOCH_MODELS:
        ests: List[Optional[Any]] = []
        classes_per_head: List[np.ndarray] = []
        for a in range(n_heads):
            y = Y_train[:, a]
            classes = np.unique(y)
            if len(classes) <= 1:
                heads.append(make_constant_head(name, y))
                ests.append(None)
                classes_per_head.append(classes)
            else:
                est = build_estimator(name, args, seed)
                if name == "sgd_log" and args.class_weight != "none":
                    est.set_params(class_weight=balanced_weight_dict(y))
                heads.append(Head(name=name, estimator=est, classes=classes))
                ests.append(est)
                classes_per_head.append(classes)

        rng = np.random.default_rng(seed)
        n_train = X_train.shape[0]
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(n_train)
            for a, est in enumerate(ests):
                if est is None:
                    continue
                y = Y_train[order, a]
                est.partial_fit(X_train[order], y, classes=classes_per_head[a])
            model = MultiHeadModel(name, heads, None, 0)
            train_loss, train_acc, train_f1 = _epoch_eval(model, X_train, Y_train)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
            }
            if X_dev is not None and Y_dev is not None:
                dev_loss, dev_acc, dev_f1 = _epoch_eval(model, X_dev, Y_dev)
                row.update(
                    {
                        "dev_loss": dev_loss,
                        "dev_accuracy": dev_acc,
                        "dev_macro_f1": dev_f1,
                    }
                )
            epoch_log.append(row)
        for head in heads:
            if head.estimator is not None:
                head.classes = np.asarray(head.estimator.classes_, dtype=int)
        return MultiHeadModel(name, heads, None, 0), epoch_log

    for a in range(n_heads):
        heads.append(_fit_one_head(name, X_train, Y_train[:, a], args, seed))
    return MultiHeadModel(name, heads, None, 0), epoch_log
