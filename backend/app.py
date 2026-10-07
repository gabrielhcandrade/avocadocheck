#!/usr/bin/env python3
"""
Streamlit interface for the Hass avocado lot dispatch PoC (stage [5] of the
proposal), written for people who never saw the project.

Flow
1. The user uploads photos of a lot, or builds an example lot from the public
   dataset with one click.
2. Every fruit is detected and cropped exactly as in training (pipeline.py).
3. For each task the app uses the model that was best in the cross validation
   of pipeline.py (metrics_stage.csv, metrics_days.csv): the CNN or the
   baseline saved in final_model_*.joblib.
4. The lot rule of the pipeline gives the recommendation, written as a plain
   sentence, and a "trip ruler" lets the user drag the trip length and see
   which fruits make it.
5. Every fruit can be opened to see its stage probabilities and, when the CNN
   is used, the Grad-CAM map.

Run
    python -m streamlit run app.py
    python -m streamlit run app.py -- --models-dir outputs_avocado
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402  same folder, shares the preprocessing

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Vocabulary and colours. The stage names follow the 1 to 5 scale of the
# dataset article; the colours go from green skin to ripe aubergine skin.
# ---------------------------------------------------------------------------

STAGE_NAMES = {
    1: "Verde e firme",
    2: "Começando a amadurecer",
    3: "Amadurecendo",
    4: "Pronto para comer",
    5: "Passando do ponto",
}
STAGE_COLOURS = {1: "#4C7A27", 2: "#7F9A2E", 3: "#A08A35", 4: "#6E4A3A", 5: "#3B2231"}

CONDITION_LABELS = {
    "T10": "Câmara fria (10 °C)",
    "T20": "Sala climatizada (20 °C)",
    "Tam": "Temperatura ambiente",
}

DESTINATIONS = {
    "export_sea": ("🚢", "Pode ir para exportação marítima", "export"),
    "domestic_distant": ("🚚", "Melhor mandar para o mercado interno", "domestic"),
    "local_market": ("🏪", "Venda local, nos próximos dias", "local"),
    "processing": ("🥑", "Encaminhar para processamento", "processing"),
    "human_review": ("🔍", "Precisa de um olhar humano", "review"),
}

CHART_FONT = "Atkinson Hyperlegible, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"

STATUS = {
    "ok": ("Aguenta", "#3F5F24"),
    "tight": ("No limite", "#B7791F"),
    "no": ("Não aguenta", "#46283A"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&display=swap');

:root {
  --ink: #1F2617;
  --muted: #5B6450;
  --paper: #F5F6EF;
  --surface: #FFFFFF;
  --line: #DDE1D0;
  --skin: #3F5F24;
  --flesh: #EEF0C4;
  --pit: #46283A;
  --amber: #B7791F;
}

.stApp { background: var(--paper); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp [data-testid="stMarkdownContainer"] {
  font-family: 'Atkinson Hyperlegible', system-ui, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: 'Bricolage Grotesque', system-ui, sans-serif;
  color: var(--ink);
  letter-spacing: -0.015em;
}
.block-container { max-width: 1180px; padding-top: 2.2rem; }

.hero .hero__title {
  font-family: 'Bricolage Grotesque', system-ui, sans-serif; color: var(--ink); letter-spacing: -0.02em;
  font-size: clamp(2.2rem, 5vw, 3.6rem); font-weight: 800; line-height: 1.02; margin: 0 0 .9rem; max-width: 18ch;
}
.hero p:not(.hero__title) { font-size: 1.15rem; color: var(--muted); max-width: 46ch; margin: 0; line-height: 1.5; }

.verdict {
  display: grid; grid-template-columns: auto 1fr; gap: 1.4rem; align-items: start;
  background: var(--surface); border: 1px solid var(--line); border-left: 10px solid var(--skin);
  border-radius: 18px; padding: 1.6rem 1.8rem; margin: .4rem 0 1.4rem;
}
.verdict--domestic { border-left-color: #6E8B2E; }
.verdict--local { border-left-color: var(--amber); }
.verdict--processing { border-left-color: var(--pit); }
.verdict--review { border-left-color: var(--amber); background: #FFFBEF; }
.verdict__icon {
  font-size: 2.6rem; width: 4.4rem; height: 4.4rem; border-radius: 50%;
  display: grid; place-items: center; background: var(--flesh);
}
.verdict__lead { color: var(--muted); margin: 0 0 .2rem; font-size: 1rem; }
.verdict h2 { font-size: clamp(1.6rem, 3vw, 2.3rem); font-weight: 800; margin: 0 0 .5rem; line-height: 1.1; }
.verdict__why { font-size: 1.1rem; line-height: 1.55; margin: 0 0 1rem; max-width: 62ch; color: var(--ink); }
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: .35rem; }
.pill { width: 22px; height: 30px; border-radius: 11px 11px 13px 13px / 14px 14px 16px 16px; }
.pills__caption { font-size: .9rem; color: var(--muted); margin: 0; }

.legend { display: flex; flex-wrap: wrap; gap: .4rem 1.1rem; margin: .2rem 0 .6rem; font-size: .95rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: .4rem; }
.legend i { width: 12px; height: 16px; border-radius: 6px; display: inline-block; }

.tally { font-size: 1.25rem; margin: .2rem 0 .8rem; color: var(--ink); }
.tally b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.6rem; }

.chip {
  display: inline-block; padding: .18rem .65rem; border-radius: 999px;
  font-size: .85rem; font-weight: 700; color: #fff;
}
.fruit-days { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; font-weight: 800; line-height: 1; margin: .35rem 0 .1rem; }
.fruit-meta { color: var(--muted); font-size: .92rem; margin: 0; }

.step { display: grid; grid-template-columns: 2.4rem 1fr; gap: .9rem; margin-bottom: 1.1rem; }
.step__n {
  width: 2.4rem; height: 2.4rem; border-radius: 50%; background: var(--skin); color: #fff;
  display: grid; place-items: center; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
}
.step h4 { margin: .1rem 0 .2rem; font-size: 1.15rem; }
.step p { margin: 0; color: var(--muted); line-height: 1.5; max-width: 62ch; }

@media (max-width: 640px) {
  .block-container { padding-top: 3.4rem; }
  .verdict { grid-template-columns: 1fr; padding: 1.2rem; gap: .8rem; }
  .verdict__icon { width: 3.4rem; height: 3.4rem; font-size: 2rem; }
}
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-radius: 16px; }
.stButton > button:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }
</style>
"""


# ---------------------------------------------------------------------------
# Models (no Streamlit here, so the functions can be tested headless)
# ---------------------------------------------------------------------------

def best_row(models_dir: Path, task: str, cnn_only: Optional[bool] = None) -> Optional[Dict]:
    """Best non baseline row of metrics_<task>.csv, optionally only CNN or only non CNN rows."""
    path = models_dir / f"metrics_{task}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[~df["model"].str.startswith("baseline")]
    if cnn_only is not None:
        df = df[df["model"].str.startswith("cnn_") == cnn_only]
    if df.empty:
        return None
    if task == "stage":
        df = df.sort_values("pooled_macro_F1", ascending=False)
    else:
        df = df.sort_values("pooled_MAE")
    return df.iloc[0].to_dict()


def load_models(models_dir: Path, device: str = "cpu") -> Dict:
    """
    Load what pipeline.py saved and choose, for each task, the model that was
    best in the cross validation. Without metrics files the CNN is preferred.
    """
    import joblib

    models: Dict = {"cnn": None, "stage_sk": None, "days_sk": None}
    cnn_path = models_dir / "cnn_final.pt"
    if cnn_path.exists() and P.HAS_TORCH:
        import torch

        ckpt = torch.load(cnn_path, map_location=device, weights_only=False)
        n_cond = len(ckpt["conditions"]) if ckpt.get("use_condition") else 0
        net = P.MultiTaskNet(ckpt["backbone"], len(P.STAGES), n_cond, pretrained=False)
        net.load_state_dict(ckpt["state_dict"])
        net.to(device).eval()
        models["cnn"] = {"net": net, "meta": ckpt, "device": device}
    for key, name in (("stage_sk", "final_model_stage.joblib"), ("days_sk", "final_model_days.joblib")):
        if (models_dir / name).exists():
            models[key] = joblib.load(models_dir / name)

    cnn = models["cnn"]
    cnn_days_ok = cnn is not None and cnn["meta"]["mode"] == "multitask"
    choice, quality = {}, {}
    for task, sk_key, cnn_ok in (("stage", "stage_sk", cnn is not None), ("days", "days_sk", cnn_days_ok)):
        overall = best_row(models_dir, task)
        prefer_cnn = cnn_ok and (overall is None or str(overall["model"]).startswith("cnn_")
                                 or models[sk_key] is None)
        if prefer_cnn:
            choice[task] = "cnn"
            quality[task] = best_row(models_dir, task, cnn_only=True)
        elif models[sk_key] is not None:
            choice[task] = "sk"
            quality[task] = best_row(models_dir, task, cnn_only=False)
        else:
            raise FileNotFoundError(
                f"Nenhum modelo de {'estágio' if task == 'stage' else 'dias'} em {models_dir}. "
                "Rode o pipeline.py antes (com --fit-final-cnn para salvar a rede neural).")
    models["choice"], models["quality"] = choice, quality

    if cnn is not None:
        meta = cnn["meta"]
        models["crop_pad"], models["crop_size"] = meta.get("crop_pad", 0.08), meta.get("crop_size", 320)
        models["conditions"] = list(meta["conditions"])
    else:
        cfg = models["stage_sk"]["config"]
        models["crop_pad"], models["crop_size"] = float(cfg.get("crop_pad", 0.08)), int(cfg.get("crop_size", 320))
        models["conditions"] = ["T10", "T20", "Tam"]

    def uses_condition(task: str) -> bool:
        if choice[task] == "cnn":
            return bool(cnn["meta"].get("use_condition"))
        return any(c.startswith("cond_") for c in models[f"{task}_sk"]["columns"])

    models["uses_condition"] = {t: uses_condition(t) for t in ("stage", "days")}
    return models


def describe_model(models: Dict, task: str) -> str:
    if models["choice"][task] == "cnn":
        return f"rede neural {models['cnn']['meta']['backbone']}"
    sk = models[f"{task}_sk"]
    name = str(sk["model"].steps[-1][1].__class__.__name__)
    return {"RandomForestRegressor": "Random Forest", "RandomForestClassifier": "Random Forest",
            "Ridge": "regressão linear", "LogisticRegression": "regressão logística"}.get(name, name) \
        + " sobre cor e textura"


def load_detector(kind: str, weights: Optional[str], prompts: List[str]):
    """YOLO model, or None for the threshold detector (also the fallback)."""
    if kind not in ("yolo-world", "yolo"):
        return None
    cfg = SimpleNamespace(detector=kind, yolo_weights=weights or None, yolo_prompts=prompts)
    return P.build_yolo(cfg)


# ---------------------------------------------------------------------------
# Detection of every fruit in a photo
# ---------------------------------------------------------------------------

def threshold_boxes(img: Image.Image, min_area: float) -> List[Tuple[np.ndarray, float]]:
    """All fruit-like components on a light background, one box per fruit."""
    w, h = img.size
    factor = min(1.0, 500.0 / max(w, h))
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    hsv = np.asarray(small.convert("HSV"))
    mask = (hsv[..., 1] >= P.MASK_MIN_SATURATION) | (hsv[..., 2] <= P.MASK_MAX_VALUE)
    mask = ndi.binary_fill_holes(ndi.binary_opening(mask, iterations=2))
    labels, n = ndi.label(mask)
    boxes = []
    for idx, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None or (labels[sl] == idx).sum() / float(mask.size) < min_area:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        boxes.append((np.array([x0, y0, x1, y1], float) / factor, float("nan")))
    return boxes


def yolo_all_boxes(model, img: Image.Image, conf: float, min_area: float) -> List[Tuple[np.ndarray, float]]:
    res = model.predict(source=np.asarray(img)[..., ::-1], conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    h, w = res.orig_shape[:2]
    out = []
    for box, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        area = (box[2] - box[0]) * (box[3] - box[1]) / float(w * h)
        if min_area <= area <= 0.98:
            out.append((box, float(c)))
    return out


def detect_fruits(img: Image.Image, detector, conf: float, min_area: float) -> Tuple[List, str]:
    if detector is not None:
        try:
            boxes = yolo_all_boxes(detector, img, conf, min_area)
        except Exception:
            boxes = []
        if boxes:
            return boxes, "yolo"
    boxes = threshold_boxes(img, min_area)
    if boxes:
        return boxes, "threshold"
    w, h = img.size
    return [(np.array([0, 0, w, h], float), float("nan"))], "full_image"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def features_for_crop(crop: Image.Image, columns: List[str], condition: str) -> pd.DataFrame:
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    buf.seek(0)
    feats = P.crop_features(buf) or {}
    for col in columns:
        if col.startswith("cond_"):
            feats[col] = 1.0 if col == f"cond_{condition}" else 0.0
    return pd.DataFrame([{c: feats.get(c, np.nan) for c in columns}]).fillna(0.0)


def predict_crops(crops: List[Image.Image], models: Dict, condition: str, with_cam: bool = True) -> pd.DataFrame:
    """Stage probabilities, remaining days, confidence and Grad-CAM (CNN only) for every crop."""
    rows = []
    cnn = models["cnn"]
    use_cnn = cnn is not None and "cnn" in models["choice"].values()
    for crop in crops:
        rec: Dict = {"cam_stage": None}
        probs = days = None
        if use_cnn:
            import torch

            meta, net, device = cnn["meta"], cnn["net"], cnn["device"]
            x = P.make_transforms(meta["img_size"], train=False)(crop).unsqueeze(0).to(device)
            n_cond = len(meta["conditions"]) if meta.get("use_condition") else 0
            cond = torch.zeros(1, n_cond, device=device)
            if n_cond and condition in meta["conditions"]:
                cond[0, meta["conditions"].index(condition)] = 1.0
            with torch.no_grad():
                logits, reg = net(x, cond)
            if models["choice"]["stage"] == "cnn":
                probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
                if with_cam:
                    rec["cam_stage"] = P.grad_cam(net, x, cond, "stage", int(probs.argmax()))
            if models["choice"]["days"] == "cnn":
                days = max(0.0, float(reg.item()) * meta["y_scale"])
        if probs is None:
            sk = models["stage_sk"]
            probs = P.predict_proba_aligned(sk["model"], features_for_crop(crop, sk["columns"], condition))[0]
        if days is None:
            sk = models["days_sk"]
            days = max(0.0, float(sk["model"].predict(features_for_crop(crop, sk["columns"], condition))[0]))
        rec.update({f"p_stage_{s}": float(p) for s, p in zip(P.STAGES, probs)})
        rec["stage_pred"] = int(np.argmax(probs)) + 1
        rec["confidence"] = float(np.max(probs))
        rec["days_pred"] = float(days)
        rows.append(rec)
    return pd.DataFrame(rows)


def analyse_photos(images: List[Tuple[str, Image.Image]], models: Dict, detector, condition: str,
                   conf: float = 0.05, min_area: float = 0.01, with_cam: bool = True
                   ) -> Tuple[pd.DataFrame, List[Image.Image]]:
    """Detect, crop and predict every fruit of every photo. The lot rule is applied later."""
    fruits, crops = [], []
    for name, img in images:
        img = img.convert("RGB")
        boxes, method = detect_fruits(img, detector, conf, min_area)
        for k, (box, det_conf) in enumerate(boxes):
            crops.append(P.square_crop(img, box, models["crop_pad"], models["crop_size"]))
            fruits.append({"photo": name, "fruit_in_photo": k + 1, "detector": method,
                           "detect_conf": det_conf, "box": box})
    if not fruits:
        return pd.DataFrame(), []
    df = pd.concat([pd.DataFrame(fruits), predict_crops(crops, models, condition, with_cam)], axis=1)
    df.insert(0, "fruit", np.arange(1, len(df) + 1))
    return df, crops


def lot_decision(df: pd.DataFrame, rule: SimpleNamespace) -> str:
    return P.decide_destination(df["days_pred"].to_numpy(), df["stage_pred"].to_numpy(),
                                df["confidence"].to_numpy(), rule)


def fruit_status(days: float, trip: float, margin: float) -> str:
    if days >= trip + margin:
        return "ok"
    if days >= trip:
        return "tight"
    return "no"


def draw_boxes(img: Image.Image, fruits: pd.DataFrame) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(3, out.size[0] // 220)
    try:
        font = ImageFont.load_default(size=max(16, out.size[0] // 28))
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    for _, f in fruits.iterrows():
        colour = STAGE_COLOURS[int(f["stage_pred"])]
        x1, y1, x2, y2 = f["box"]
        draw.rounded_rectangle([x1, y1, x2, y2], radius=width * 4, outline=colour, width=width)
        label = f"{f['fruit']}"
        tb = draw.textbbox((0, 0), label, font=font)
        pad = width * 2
        draw.rounded_rectangle([x1, y1, x1 + tb[2] + 2 * pad, y1 + tb[3] + 2 * pad], radius=width * 3, fill=colour)
        draw.text((x1 + pad, y1 + pad), label, fill="white", font=font)
    return out


def overlay_cam(crop: Image.Image, cam: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm

    heat = Image.fromarray((cm.magma(cam)[..., :3] * 255).astype(np.uint8)).resize(crop.size)
    return Image.blend(crop.convert("RGB"), heat, 0.5)


def example_lot(dataset_dir: Path, condition: str, n: int, seed: int) -> List[Tuple[str, bytes]]:
    """A mixed lot from the public dataset: n different fruits, each on a random day."""
    files = [p for p in dataset_dir.rglob(f"{condition}_*_a_*.jpg")]
    by_fruit: Dict[str, List[Path]] = {}
    for p in files:
        parsed = P.parse_file_name(p.stem)
        if parsed:
            by_fruit.setdefault(parsed["sample"], []).append(p)
    rng = random.Random(seed)
    chosen = rng.sample(sorted(by_fruit), k=min(n, len(by_fruit)))
    return [(p.name, p.read_bytes()) for p in (rng.choice(by_fruit[f]) for f in chosen)]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=HERE / "outputs_avocado")
    parser.add_argument("--dataset-dir", type=Path, default=HERE / "Hass Avocado Ripening Photographic Dataset")
    args, _ = parser.parse_known_args()
    return args


def fmt_days(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def verdict_html(destination: str, df: pd.DataFrame, rule: SimpleNamespace) -> str:
    icon, title, css = DESTINATIONS[destination]
    worst_share = round(100 * (1 - rule.lot_quantile))
    q = float(np.percentile(df["days_pred"], 100 * (1 - rule.lot_quantile)))
    need_exp = rule.export_days + rule.safety_margin
    need_dom = rule.domestic_days + rule.safety_margin
    n = len(df)
    n_exp = int((df["days_pred"] >= need_exp).sum())
    if destination == "export_sea":
        why = (f"{n_exp} de {n} frutos têm pelo menos {fmt_days(need_exp)} dias de vida verde. "
               f"Mesmo os {worst_share}% mais maduros ainda têm {fmt_days(q)} dias, "
               "o bastante para a viagem de navio com folga.")
    elif destination == "domestic_distant":
        why = (f"Os {worst_share}% mais maduros têm só {fmt_days(q)} dias, menos que os {fmt_days(need_exp)} "
               f"que o navio exige, mas o suficiente para os {fmt_days(need_dom)} dias de caminhão.")
    elif destination == "local_market":
        why = (f"Os {worst_share}% mais maduros têm só {fmt_days(q)} dias de vida verde. "
               "É pouco para viajar longe: vale vender perto.")
    elif destination == "processing":
        share = float((df["stage_pred"] >= 5).mean())
        why = f"{share:.0%} dos frutos já passaram do ponto de consumo. Ainda servem para polpa, guacamole ou óleo."
    else:
        why = (f"O sistema ficou em dúvida: a certeza média foi de {df['confidence'].mean():.0%}. "
               "Confira os frutos abaixo antes de decidir o destino.")
    pills = "".join(
        f'<span class="pill" title="Fruto {r.fruit}: {STAGE_NAMES[r.stage_pred]}" '
        f'style="background:{STAGE_COLOURS[r.stage_pred]}"></span>'
        for r in df.sort_values("days_pred", ascending=False).itertuples()
    )
    return (
        f'<div class="verdict verdict--{css}"><div class="verdict__icon">{icon}</div><div>'
        f'<p class="verdict__lead">Recomendação para este lote de {n} frutos</p>'
        f"<h2>{title}</h2><p class=\"verdict__why\">{why}</p>"
        f'<div class="pills">{pills}</div>'
        '<p class="pills__caption">Cada forma é um fruto, do mais verde ao mais maduro.</p>'
        "</div></div>"
    )


def legend_html() -> str:
    items = "".join(f'<span><i style="background:{STAGE_COLOURS[s]}"></i>{s}. {STAGE_NAMES[s]}</span>'
                    for s in P.STAGES)
    return f'<div class="legend">{items}</div>'


def trip_ruler(df: pd.DataFrame, trip: float, margin: float):
    """Beeswarm of the remaining days, with the trip length and the safety margin."""
    import altair as alt

    plot = df[["fruit", "days_pred", "stage_pred", "confidence"]].copy()
    plot["estagio"] = plot["stage_pred"].map(lambda s: f"{s}. {STAGE_NAMES[s]}")
    plot["dias"] = plot["days_pred"].round(1)
    plot["certeza"] = (plot["confidence"] * 100).round(0).astype(int).astype(str) + "%"
    plot["bin"] = (plot["days_pred"] * 2).round() / 2
    plot["lane"] = plot.sort_values("days_pred").groupby("bin").cumcount()
    xmax = max(float(plot["days_pred"].max()), trip + margin) + 1.5
    n_lanes = int(plot["lane"].max()) + 1

    pick = alt.selection_point(fields=["fruit"], name="fruta", on="click")
    band = alt.Chart(pd.DataFrame({"a": [trip], "b": [trip + margin]})).mark_rect(
        color="#B7791F", opacity=0.14).encode(x="a:Q", x2="b:Q")
    line = alt.Chart(pd.DataFrame({"a": [trip]})).mark_rule(color="#1F2617", strokeWidth=2, strokeDash=[6, 4]).encode(x="a:Q")
    label = alt.Chart(pd.DataFrame({"a": [trip], "t": [f"viagem de {fmt_days(trip)} dias"]})).mark_text(
        align="left", dx=6, dy=-8, color="#1F2617", fontSize=13, fontWeight="bold", font=CHART_FONT).encode(
        x="a:Q", y=alt.value(0), text="t:N")
    points = alt.Chart(plot).mark_circle(size=380, opacity=1, stroke="white", strokeWidth=2).encode(
        x=alt.X("days_pred:Q", title="Dias de vida verde que ainda restam",
                scale=alt.Scale(domain=[0, xmax], nice=False), axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("lane:Q", axis=None, scale=alt.Scale(domain=[-0.8, n_lanes - 0.2])),
        color=alt.Color("estagio:N", legend=None,
                        scale=alt.Scale(domain=[f"{s}. {STAGE_NAMES[s]}" for s in P.STAGES],
                                        range=[STAGE_COLOURS[s] for s in P.STAGES])),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.35)),
        tooltip=[alt.Tooltip("fruit:N", title="Fruto"), alt.Tooltip("estagio:N", title="Estágio"),
                 alt.Tooltip("dias:Q", title="Dias restantes"), alt.Tooltip("certeza:N", title="Certeza")],
    ).add_params(pick)
    num = alt.Chart(plot).mark_text(color="white", fontSize=11, fontWeight="bold", font=CHART_FONT).encode(
        x="days_pred:Q", y="lane:Q", text="fruit:N")
    return (band + line + label + points + num).properties(height=90 + 34 * n_lanes).configure_view(
        strokeWidth=0).configure_axis(labelFont=CHART_FONT, titleFont=CHART_FONT,
                                      labelFontSize=13, titleFontSize=14, titleColor="#5B6450",
                                      labelColor="#5B6450", domainColor="#DDE1D0", tickColor="#DDE1D0")


def stage_probability_chart(row: pd.Series):
    import altair as alt

    data = pd.DataFrame({
        "estagio": [f"{s}. {STAGE_NAMES[s]}" for s in P.STAGES],
        "prob": [row[f"p_stage_{s}"] for s in P.STAGES],
    })
    data["txt"] = (data["prob"] * 100).round(0).astype(int).astype(str) + "%"
    base = alt.Chart(data).encode(
        y=alt.Y("estagio:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=13)),
        x=alt.X("prob:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 1.15])),
    )
    bars = base.mark_bar(cornerRadiusEnd=6, height=18).encode(
        color=alt.Color("estagio:N", legend=None, scale=alt.Scale(
            domain=list(data["estagio"]), range=[STAGE_COLOURS[s] for s in P.STAGES])))
    text = base.mark_text(align="left", dx=5, fontSize=13, color="#1F2617", font=CHART_FONT).encode(text="txt:N")
    return (bars + text).properties(height=190).configure_view(strokeWidth=0).configure_axis(
        labelFont=CHART_FONT, labelColor="#1F2617", domain=False, ticks=False)


def run_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Para onde vai este lote?", page_icon="🥑", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    cli = parse_cli()
    ss = st.session_state

    @st.cache_resource(show_spinner="Carregando os modelos...")
    def cached_models(path: str):
        return load_models(Path(path))

    @st.cache_resource(show_spinner="Carregando o detector de frutos...")
    def cached_detector(kind: str, weights: str, prompts: str):
        return load_detector(kind, weights, [p.strip() for p in prompts.split(",") if p.strip()])

    @st.cache_data(show_spinner=False, max_entries=6)
    def cached_analysis(_models, _detector, key: str, files: Tuple[Tuple[str, bytes], ...],
                        condition: str, conf: float, min_area: float):
        images = [(name, Image.open(io.BytesIO(data))) for name, data in files]
        return analyse_photos(images, _models, _detector, condition, conf, min_area, with_cam=True)

    # ----- advanced settings, hidden in the sidebar --------------------------
    with st.sidebar:
        st.header("Configurações avançadas")
        st.caption("Os valores padrão são os mesmos do pipeline. Mude só se souber o que está fazendo.")
        models_dir = st.text_input("Pasta dos modelos", str(cli.models_dir))
        dataset_dir = st.text_input("Pasta do dataset (para o lote de exemplo)", str(cli.dataset_dir))
        st.subheader("Regra do lote")
        export_days = st.number_input("Dias de viagem até o importador (navio)", 0.0, 60.0, 10.0, 1.0)
        domestic_days = st.number_input("Dias de viagem no mercado interno (caminhão)", 0.0, 30.0, 4.0, 1.0)
        margin = st.number_input("Margem de segurança (dias)", 0.0, 10.0, 2.0, 0.5)
        quantile = st.slider("Parte do lote que precisa aguentar", 0.5, 1.0, 0.9, 0.05,
                             help="0,9 significa que a decisão olha para os 10% de frutos mais maduros.")
        review = st.slider("Certeza mínima antes de pedir revisão humana", 0.0, 1.0, 0.6, 0.05)
        st.subheader("Detector de frutos")
        det_kind = st.selectbox("Tipo", ["yolo-world", "yolo", "threshold"],
                                help="Se o YOLO não achar nada, o detector por cor (threshold) assume.")
        det_weights = st.text_input("Pesos YOLO (opcional)", "") if det_kind != "threshold" else ""
        det_prompts = st.text_input("Texto do YOLO-World", "avocado") if det_kind == "yolo-world" else "avocado"
        det_conf = st.slider("Confiança mínima da caixa", 0.01, 0.9, 0.05, 0.01)
        min_area = st.slider("Tamanho mínimo do fruto na foto", 0.002, 0.2, 0.01, 0.002)

    # ----- hero ----------------------------------------------------------------
    st.markdown(
        '<div class="hero"><p class="hero__title" role="heading" aria-level="1">Para onde vai este lote de abacates?</p>'
        "<p>Envie fotos dos frutos. O sistema estima quantos dias de vida verde cada um ainda tem "
        "e diz se o lote aguenta a viagem de navio, de caminhão ou se deve ser vendido perto.</p></div>",
        unsafe_allow_html=True)
    st.write("")

    try:
        models = cached_models(models_dir)
    except Exception as exc:
        st.error(f"Não consegui carregar os modelos: {exc}")
        st.info("Abra o menu lateral (seta no canto superior esquerdo) e confira a pasta dos modelos.")
        st.stop()
    rule = SimpleNamespace(export_days=export_days, domestic_days=domestic_days, safety_margin=margin,
                           lot_quantile=quantile, review_confidence=review)


    tab_lote, tab_como = st.tabs(["Analisar um lote", "Como funciona"])

    with tab_como:
        render_how_it_works(st, models)

    with tab_lote:
        left, right = st.columns([3, 2], gap="large")
        with left:
            uploads = st.file_uploader("Fotos do lote", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                       help="Uma foto por fruto ou vários frutos na mesma foto, de preferência "
                                            "sobre fundo claro.")
        with right:
            options = [c for c in models["conditions"] if c in CONDITION_LABELS] or models["conditions"]
            condition = st.radio("Onde os frutos ficaram guardados?", options,
                                 format_func=lambda c: CONDITION_LABELS.get(c, c), horizontal=False,
                                 help="O abacate amadurece mais rápido no calor. Os dias previstos valem "
                                      "para a condição escolhida.")
            if not models["uses_condition"]["days"]:
                st.caption("O modelo de dias atual não usa essa informação.")
            ds_path = Path(dataset_dir)
            if ds_path.exists():
                if st.button("Montar um lote de exemplo", width="stretch",
                             help="Sorteia 15 frutos diferentes do dataset público, cada um num dia diferente."):
                    ss["seed"] = ss.get("seed", 0) + 1
                    ss["example"] = example_lot(ds_path, condition, 15, ss["seed"])
                    ss.pop("fruit_sel", None)

        if uploads:
            files = tuple((f.name, f.getvalue()) for f in uploads)
            source = "upload"
        elif ss.get("example"):
            files = tuple(ss["example"])
            source = "example"
        else:
            st.info("Envie as fotos do lote ou monte um lote de exemplo para começar.")
            return

        key = f"{models_dir}|{det_kind}|{det_weights}|{det_prompts}"
        fingerprint = hashlib.md5(b"".join(d for _, d in files) + condition.encode()).hexdigest()
        if ss.get("fingerprint") != fingerprint:
            ss["fingerprint"] = fingerprint
            ss.pop("fruit_sel", None)
        detector = cached_detector(det_kind, det_weights, det_prompts) if det_kind != "threshold" else None
        with st.spinner(f"Procurando os frutos em {len(files)} foto(s) e lendo a casca de cada um..."):
            df, crops = cached_analysis(models, detector, key, files, condition, det_conf, min_area)
        if df.empty:
            st.warning("Não encontrei nenhum abacate nas fotos. Tente uma foto com o fruto inteiro e fundo claro.")
            return
        if source == "example":
            st.caption(f"Lote de exemplo com {len(files)} frutos do dataset público, guardados em "
                       f"{CONDITION_LABELS.get(condition, condition).lower()}.")

        destination = lot_decision(df, rule)
        st.markdown(verdict_html(destination, df, rule), unsafe_allow_html=True)

        # ----- interactive trip ruler ------------------------------------------
        st.subheader("Simule a viagem")
        presets = {
            f"🚢 Navio ({fmt_days(export_days)} dias)": export_days,
            f"🚚 Caminhão ({fmt_days(domestic_days)} dias)": domestic_days,
            "🏪 Venda local (1 dia)": 1.0,
            "Outro tempo": None,
        }
        control = getattr(st, "segmented_control", None)
        if control is not None:
            choice = control("Qual viagem você quer testar?", list(presets), default=list(presets)[0],
                             key="trip_choice") or list(presets)[0]
        else:
            choice = st.radio("Qual viagem você quer testar?", list(presets), horizontal=True, key="trip_choice")
        trip = presets[choice]
        if trip is None:
            trip = st.slider("Dias de viagem", 0.0, 30.0, float(export_days), 0.5)

        df["status"] = [fruit_status(d, trip, margin) for d in df["days_pred"]]
        counts = df["status"].value_counts()
        st.markdown(
            f'<p class="tally"><b>{counts.get("ok", 0)}</b> de {len(df)} frutos aguentam com folga, '
            f'<b>{counts.get("tight", 0)}</b> chegam no limite e <b>{counts.get("no", 0)}</b> não aguentam.</p>',
            unsafe_allow_html=True)

        chart_col, detail_col = st.columns([3, 2], gap="large")
        with chart_col:
            st.markdown(legend_html(), unsafe_allow_html=True)
            event = st.altair_chart(trip_ruler(df, trip, margin), width="stretch",
                                    on_select="rerun", key=f"ruler_{fingerprint}")
            st.caption(f"A linha tracejada é a viagem; a faixa clara são os {fmt_days(margin)} dias de margem "
                       "de segurança. Clique num fruto para ver os detalhes.")
            picked = (event.selection.get("fruta") if event and getattr(event, "selection", None) else None) or []
            if picked:
                ss["fruit_sel"] = int(picked[0]["fruit"])
        with detail_col:
            sel = ss.get("fruit_sel", int(df.sort_values("days_pred").iloc[0]["fruit"]))
            render_fruit_detail(st, df, crops, sel, trip, margin, models)

        # ----- all fruits ---------------------------------------------------------
        st.subheader("Todos os frutos")
        order = st.radio("Ordenar por", ["Mais maduros primeiro", "Ordem das fotos"], horizontal=True,
                         label_visibility="collapsed")
        view = df.sort_values("days_pred") if order == "Mais maduros primeiro" else df
        cols = st.columns(5)
        for i, (_, f) in enumerate(view.iterrows()):
            idx = int(f["fruit"]) - 1
            label, colour = STATUS[f["status"]]
            with cols[i % 5]:
                with st.container(border=True):
                    st.image(crops[idx], width="stretch")
                    st.markdown(
                        f'<span class="chip" style="background:{colour}">{label}</span>'
                        f'<p class="fruit-days">{fmt_days(f["days_pred"])} dias</p>'
                        f'<p class="fruit-meta">Fruto {f["fruit"]}, {STAGE_NAMES[int(f["stage_pred"])].lower()}</p>',
                        unsafe_allow_html=True)
                    if st.button("Ver detalhes", key=f"btn_{f['fruit']}", width="stretch"):
                        ss["fruit_sel"] = int(f["fruit"])
                        st.rerun()

        with st.expander("Ver as fotos originais com os frutos encontrados"):
            images = {name: Image.open(io.BytesIO(data)) for name, data in files}
            photo_cols = st.columns(3)
            for i, (name, img) in enumerate(images.items()):
                with photo_cols[i % 3]:
                    st.image(draw_boxes(img, df[df["photo"] == name]), caption=name, width="stretch")

        report = df.drop(columns=["box", "cam_stage"]).copy()
        report.insert(1, "condition", condition)
        report["stage_name"] = report["stage_pred"].map(STAGE_NAMES)
        report["trip_days_tested"] = trip
        report["lot_destination"] = destination
        st.download_button("Baixar relatório do lote (CSV)", report.to_csv(index=False).encode("utf-8"),
                           file_name="relatorio_lote.csv", mime="text/csv")


def days_sentence(days: float, trip: float) -> str:
    if days < 0.5:
        return "Já chegou ao ponto de consumo, então não aguenta viagem nenhuma."
    return (f"Estimamos <b>{fmt_days(days)} dias</b> até ficar pronto para comer; "
            f"a viagem testada leva {fmt_days(trip)}.")


def render_fruit_detail(st, df: pd.DataFrame, crops: List[Image.Image], fruit: int,
                        trip: float, margin: float, models: Dict) -> None:
    row = df[df["fruit"] == fruit].iloc[0]
    crop = crops[fruit - 1]
    label, colour = STATUS[row["status"]]
    with st.container(border=True):
        st.markdown(f"#### Fruto {fruit}")
        a, b = st.columns(2)
        a.image(crop, caption="Foto recortada", width="stretch")
        if row["cam_stage"] is not None:
            b.image(overlay_cam(crop, row["cam_stage"]), caption="Onde a rede olhou", width="stretch")
        else:
            b.caption("O mapa de calor só existe quando a rede neural decide o estágio.")
        stage = int(row["stage_pred"])
        st.markdown(
            f'<span class="chip" style="background:{colour}">{label}</span> '
            f"<p style='margin:.6rem 0 .2rem;font-size:1.05rem;line-height:1.5'>"
            f"Está em <b>{STAGE_NAMES[stage].lower()}</b> (estágio {stage} de 5), com "
            f"{row['confidence']:.0%} de certeza. {days_sentence(row['days_pred'], trip)}</p>",
            unsafe_allow_html=True)
        st.altair_chart(stage_probability_chart(row), width="stretch")
        if row["confidence"] < 0.6:
            st.caption("Certeza baixa: vale um operador conferir este fruto.")


def render_how_it_works(st, models: Dict) -> None:
    q_stage, q_days = models["quality"].get("stage"), models["quality"].get("days")
    steps = [
        ("Encontra cada abacate", "Um detector (YOLO) localiza os frutos na foto e recorta cada um, "
                                  "do mesmo jeito que foi feito no treino."),
        ("Lê a casca", f"Para o estágio de maturação usamos {describe_model(models, 'stage')}. A cor, o brilho e a "
                       "textura da casca mudam aos poucos do verde ao quase preto."),
        ("Estima os dias que restam", f"Para os dias usamos {describe_model(models, 'days')}. Ele aprendeu com "
                                      "fotos de frutos acompanhados todos os dias até ficarem prontos para comer."),
        ("Decide pelo pior caso", "Um lote vale o que valem os seus frutos mais maduros. A decisão olha para "
                                  "os mais maduros do lote e compara com o tempo de viagem mais uma margem."),
    ]
    html = "".join(f'<div class="step"><div class="step__n">{i}</div><div><h4>{t}</h4><p>{d}</p></div></div>'
                   for i, (t, d) in enumerate(steps, start=1))
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(html, unsafe_allow_html=True)
    with c2:
        st.markdown("#### Quanto dá para confiar")
        if q_stage:
            st.write(f"Em frutos que o modelo nunca viu, ele acertou o estágio exato em "
                     f"**{q_stage['pooled_accuracy']:.0%}** dos casos e errou por no máximo um estágio em "
                     f"**{q_stage['pooled_within_1_stage']:.0%}**.")
        if q_days:
            st.write(f"Os dias restantes erram em média **{fmt_days(q_days['pooled_MAE'])} dia(s)**, e "
                     f"**{q_days['pooled_within_2_days']:.0%}** das estimativas ficam a até 2 dias do real.")
        st.write("Esses números vêm da validação cruzada do pipeline, separando os frutos de treino e de teste.")
        st.markdown("#### Cuidados")
        st.write("Os modelos foram treinados com fotos de estúdio, fundo branco e luz controlada. Fotos com "
                 "outra luz ou outro fundo podem piorar o resultado. Os dias previstos valem para a condição de "
                 "armazenamento escolhida, e os nomes dos estágios seguem a escala de 1 a 5 do dataset.")


if __name__ == "__main__":
    run_app()
