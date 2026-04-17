import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title="Confusion Matrix",
    save_path=None
):
    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = ["Negative", "Positive"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_scores = importance[indices]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_scores, y=top_features)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_metric_bar(metrics_dict, title="Model Performance Metrics", save_path=None):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, alpha=0.8)
    plt.title(title)
    plt.ylim(0, 1.05)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.01,
            f"{yval:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()