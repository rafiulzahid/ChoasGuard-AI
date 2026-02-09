import os
import time
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Optional Captum (IG)
# -------------------------
try:
    from captum.attr import LayerIntegratedGradients
    CAPTUM_OK = True
except Exception:
    CAPTUM_OK = False

# -------------------------
# Config
# -------------------------
LABEL_COLS = [
    "Vulgar-based",
    "Religious-Hostility",
    "Troll-based",
    "Insult-based",
    "Loathe-based",
    "Threat-based",
    "Race-based",
    "humiliation-based",
    "Political-Chaos",
    "Non-Toxic"
]

# ✅ Your Google Drive path (keep as raw string)
MODEL_DIR = r"M:\My Drive\My Work\Chaos\Code-mixed Chaos  Multi-labeled Banglish & Bangla\Code-mixed Chaos  Multi-labeled Banglish & Bangla\trained_xlmr_model"

MAX_LENGTH_DEFAULT = 192
THRESHOLD_DEFAULT = 0.5
MC_SAMPLES_DEFAULT = 20
IG_STEPS_DEFAULT = 32

# -------------------------
# Flask
# -------------------------
app = Flask(__name__)

# -------------------------
# Drive-safe checks
# -------------------------
def ensure_drive_ready(model_dir: str, wait_sec: float = 0.5, tries: int = 10):
    """
    Google Drive (M:) can be slow / stream files. This waits until required files are accessible.
    """
    required_files = ["config.json", "model.safetensors"]
    last_err = None

    for _ in range(tries):
        try:
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"MODEL_DIR not found: {model_dir}")

            files = os.listdir(model_dir)
            missing = [f for f in required_files if f not in files]
            if missing:
                raise FileNotFoundError(f"Missing in MODEL_DIR: {missing} | Found: {files}")

            # Touch read to ensure files are actually downloadable/offline-ready
            for rf in required_files:
                fp = os.path.join(model_dir, rf)
                with open(fp, "rb") as _:
                    pass

            return  # ✅ ready

        except Exception as e:
            last_err = e
            time.sleep(wait_sec)

    raise RuntimeError(f"Google Drive path not ready after retries. Last error: {last_err}")

# -------------------------
# Load tokenizer + model (robust)
# -------------------------
print("MODEL_DIR =", MODEL_DIR)
ensure_drive_ready(MODEL_DIR)

print("MODEL_DIR exists? ", os.path.exists(MODEL_DIR))
print("MODEL_DIR files: ", os.listdir(MODEL_DIR))

# 🔥 KEY FIX: force slow tokenizer on Windows/Python3.12 to avoid TokenizerFast crash
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
print("✅ Tokenizer loaded (slow)")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
print("✅ Model loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("✅ Model moved to:", device)

# -------------------------
# Helpers
# -------------------------
def predictive_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))

def enable_dropout(m: torch.nn.Module):
    # Keep model in eval(), but activate dropout layers for MC Dropout
    for layer in m.modules():
        if layer.__class__.__name__.startswith("Dropout"):
            layer.train()

def mc_dropout(text: str, n_samples=MC_SAMPLES_DEFAULT, max_length=MAX_LENGTH_DEFAULT):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    enable_dropout(model)

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(**enc)
            probs = torch.sigmoid(out.logits).detach().cpu().numpy()
            preds.append(probs)

    preds = np.stack(preds, axis=0)      # [T, 1, L]
    mean_probs = preds.mean(axis=0)[0]   # [L]
    std_probs = preds.std(axis=0)[0]     # [L]
    return mean_probs, std_probs

def lig_explain(text: str, label_name: str, max_length=MAX_LENGTH_DEFAULT, n_steps=IG_STEPS_DEFAULT):
    """
    Integrated Gradients explanation for one label.
    Requires: pip install captum
    """
    if not CAPTUM_OK:
        raise RuntimeError("Captum is not installed. Install with: pip install captum")

    if label_name not in LABEL_COLS:
        raise ValueError("Invalid label")

    target_idx = LABEL_COLS.index(label_name)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    baseline_ids = torch.full_like(input_ids, pad_id).to(device)

    # Robust for HF models
    emb_layer = model.get_input_embeddings()

    def forward_func(ids, mask, t_idx):
        outputs = model(input_ids=ids, attention_mask=mask)
        return outputs.logits[:, t_idx]

    lig = LayerIntegratedGradients(forward_func, emb_layer)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask, target_idx),
        n_steps=n_steps,
        return_convergence_delta=True
    )

    token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    token_ids = input_ids.squeeze(0).detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    mask = attention_mask.squeeze(0).detach().cpu().numpy()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        prob = torch.sigmoid(out.logits)[0, target_idx].item()

    clean = []
    for tok, score, m in zip(tokens, token_attr, mask):
        if m == 1 and tok not in ["<s>", "</s>"]:
            clean.append((tok, float(score)))

    max_abs = max(1e-9, max(abs(s) for _, s in clean)) if clean else 1.0
    clean = [(t, s / max_abs) for t, s in clean]

    return prob, clean, float(delta.item())

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    # Requires templates/index.html
    return render_template("index.html", labels=LABEL_COLS)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    threshold = float(data.get("threshold", THRESHOLD_DEFAULT))
    max_length = int(data.get("max_length", MAX_LENGTH_DEFAULT))
    mc_samples = int(data.get("mc_samples", MC_SAMPLES_DEFAULT))

    mean_probs, std_probs = mc_dropout(text, n_samples=mc_samples, max_length=max_length)
    entropy = predictive_entropy(mean_probs)

    results = []
    for i, label in enumerate(LABEL_COLS):
        results.append({
            "label": label,
            "probability": float(mean_probs[i]),
            "uncertainty": float(std_probs[i]),
            "entropy": float(entropy[i]),
            "toxic": bool(mean_probs[i] >= threshold)
        })

    results_sorted = sorted(results, key=lambda r: (r["toxic"], r["probability"]), reverse=True)
    return jsonify({"results": results_sorted})

@app.route("/api/explain", methods=["POST"])
def api_explain():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    label = data.get("label")

    if not text:
        return jsonify({"error": "Empty text"}), 400
    if label not in LABEL_COLS:
        return jsonify({"error": "Invalid label"}), 400

    max_length = int(data.get("max_length", MAX_LENGTH_DEFAULT))
    ig_steps = int(data.get("ig_steps", IG_STEPS_DEFAULT))

    try:
        prob, token_scores, delta = lig_explain(text, label, max_length=max_length, n_steps=ig_steps)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "label": label,
        "probability": float(prob),
        "delta": float(delta),
        "tokens": [{"token": t, "score": float(s)} for t, s in token_scores]
    })

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # Local run: python main.py
    # Deployment: use PORT env var if set
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
