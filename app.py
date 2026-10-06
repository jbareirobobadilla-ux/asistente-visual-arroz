"""Asistente visual de arroz: entrenamiento reproducible e interfaz Streamlit.

La red preentrenada extrae características; el clasificador aprende las etiquetas.
Ejecutar la interfaz: python -m streamlit run app.py
El notebook importa estas funciones; importar este archivo no abre la interfaz.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

SEED = 42
DATASET = "Project-AgML/paddy_disease_classification"
REVISION = "e7cd4e729b637a24d3b66fbf3cf2ead4761d5c33"
SHARDS = {"train-00000-of-00002.parquet": 403335318,
          "train-00001-of-00002.parquet": 416410458}
LABELS = {3: "blast", 4: "brown_spot", 8: "normal"}
CLASSES = ["blast", "brown_spot", "normal"]
NAMES = {"blast": "Compatible con piricularia",
         "brown_spot": "Compatible con mancha parda", "normal": "Aspecto normal"}
FEATURES = [f"f_{i:03d}" for i in range(576)]
BASE = Path(__file__).resolve().parent
WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"


def download(url: str, destination: Path, expected_size: int | None = None):
    """Descarga pública, con archivo temporal y comprobación de tamaño."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Arroz-Curso-ML/1.0"})
            with urlopen(req, timeout=90) as src, partial.open("wb") as dst:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise ValueError("El tamaño descargado no coincide con la revisión fijada.")
            partial.replace(destination)
            print(f"Descargado: {destination.name}", flush=True)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def visual_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefs = dctn(np.asarray(gray, dtype=np.float32), norm="ortho")[:8, :8].ravel()
    return int.from_bytes(np.packbits(coefs > np.median(coefs[1:])).tobytes(), "big")


def prepare_data(root: Path):
    """Revisa todas las imágenes elegidas, agrupa similitudes y separa los grupos.

    Los grupos se forman antes del split. No son identificadores de planta.
    Se eliminan copias exactas redundantes, conservando un registro por SHA256.
    """
    import pyarrow.parquet as pq
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(download,
                f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}",
                root / name, size) for name, size in SHARDS.items()]
        files = [job.result() for job in jobs]
    manifest, audit_path = root / "manifest_imagenes.json", root / "auditoria_datos.json"
    if manifest.exists() and audit_path.exists():
        rows = json.loads(manifest.read_text())
        audit = json.loads(audit_path.read_text())
        if audit["revision"] != REVISION:
            raise ValueError("La auditoría corresponde a otra revisión.")
        lookup = {(r["archivo"], r["fila"]): r for r in rows}
    else:
        rows, lookup, audit = [], {}, None
    all_counts, failures = Counter(), []
    for file in files:
        index = 0
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64, columns=["image", "label"]):
            for item in batch.to_pylist():
                label, local_index = int(item["label"]), index
                index += 1
                all_counts[label] += 1
                if label not in LABELS:
                    continue
                raw = item["image"]["bytes"]
                identifier = f"{file.stem}_r{local_index:05d}"
                image_path = root / "imagenes" / LABELS[label] / f"{identifier}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                row = lookup.get((file.name, local_index))
                if row is None:
                    try:
                        with Image.open(io.BytesIO(raw)) as original:
                            im = ImageOps.exif_transpose(original).convert("RGB")
                            im.load()
                            phash, (w, h) = visual_hash(im), im.size
                        row = {"id": identifier, "clase": LABELS[label], "archivo": file.name,
                               "fila": local_index, "ruta_original": item["image"].get("path"),
                               "ancho": w, "alto": h, "sha256": hashlib.sha256(raw).hexdigest(),
                               "phash": f"{phash:016x}"}
                        rows.append(row)
                    except Exception as exc:
                        failures.append({"id": identifier, "error": str(exc)})
                        continue
                if not image_path.exists():
                    image_path.write_bytes(raw)
        print(f"Revisado: {file.name}", flush=True)
    if audit is None:
        parent = list(range(len(rows)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            x, y = find(i), find(j)
            if x != y:
                parent[y] = x
        seen, duplicates, pairs = {}, 0, 0
        hashes = [int(r["phash"], 16) for r in rows]
        for i, row in enumerate(rows):
            if row["sha256"] in seen:
                union(i, seen[row["sha256"]])
                duplicates += 1
            else:
                seen[row["sha256"]] = i
        for i in range(len(rows)):
            for j in range(i):
                if (hashes[i] ^ hashes[j]).bit_count() <= 4:
                    union(i, j)
                    pairs += 1
        for i, row in enumerate(rows):
            row["grupo"] = f"g{find(i):05d}"
        conflicts = pd.DataFrame(rows).groupby("grupo")["clase"].nunique()
        audit = {"dataset": DATASET, "revision": REVISION,
                 "imagenes_totales": sum(all_counts.values()),
                 "clases_elegidas": dict(Counter(r["clase"] for r in rows)),
                 "imagenes_utilizables": len(rows), "imagenes_no_legibles": failures,
                 "coincidencias_exactas": duplicates,
                 "pares_visualmente_similares_phash_distancia_maxima_4": pairs,
                 "grupos": len(set(r["grupo"] for r in rows)),
                 "grupos_con_multiples_clases_requieren_revision": list(conflicts[conflicts > 1].index),
                 "limite": "No hay identificadores de planta, parcela ni sesión. La agrupación visual no garantiza independencia biológica."}
        manifest.write_text(json.dumps(rows, ensure_ascii=False))
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["grupos_con_multiples_clases_requieren_revision"]:
        raise ValueError("Hay grupos con etiquetas diferentes: revisar antes de entrenar.")
    data = pd.DataFrame(rows).drop_duplicates("sha256", keep="first").reset_index(drop=True)
    data["imagen_path"] = [str(root / "imagenes" / r.clase / f"{r.id}.jpg") for r in data.itertuples()]
    groups = data[["grupo", "clase"]].drop_duplicates().sort_values("grupo")
    training, other = train_test_split(groups, test_size=0.30, random_state=SEED, stratify=groups["clase"])
    validation, testing = train_test_split(other, test_size=0.50, random_state=SEED, stratify=other["clase"])
    mapping = {g: split for frame, split in [(training, "train"), (validation, "val"), (testing, "test")]
               for g in frame["grupo"]}
    data["split"] = data["grupo"].map(mapping)
    assert data.groupby("grupo")["split"].nunique().max() == 1
    assert data.groupby("sha256")["split"].nunique().max() == 1
    data.to_csv(root / "particiones.csv", index=False)
    return data, audit


def make_extractor(root: Path | None = None, state: dict | None = None):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    net = mobilenet_v3_small(weights=None)
    if state is None:
        path = download(WEIGHTS_URL, Path(root) / "mobilenet_v3_small-047dcff4.pth", 10306551)
        if not hashlib.sha256(path.read_bytes()).hexdigest().startswith("047dcff4"):
            raise ValueError("No coincide la firma de los pesos oficiales.")
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.classifier = torch.nn.Identity()
    else:
        net.classifier = torch.nn.Identity()
        net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()}, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    return net, transform


def extract_features(data: pd.DataFrame, root: Path, batch_size=32):
    root = Path(root)
    cache = root / "caracteristicas.npz"
    net, transform = make_extractor(root)
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if list(saved["ids"]) == list(data["id"]) and str(saved["revision"]) == REVISION:
            return pd.DataFrame(saved["X"], columns=FEATURES), net
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            tensors = []
            for path in data.iloc[start:start + batch_size]["imagen_path"]:
                with Image.open(path) as im:
                    tensors.append(transform(ImageOps.exif_transpose(im).convert("RGB")))
            features = net(torch.stack(tensors)).cpu().numpy().astype(np.float32)
            blocks.append(features)
            if start % (batch_size * 10) == 0:
                print(f"Características: {min(start + batch_size, len(data))}/{len(data)}", flush=True)
    X = np.concatenate(blocks)
    assert X.shape == (len(data), len(FEATURES)) and np.isfinite(X).all()
    np.savez_compressed(cache, X=X, ids=np.asarray(data["id"], dtype=str), revision=REVISION)
    return pd.DataFrame(X, columns=FEATURES), net


def make_pipeline(estimator):
    numeric = Pipeline([("imputacion", SimpleImputer(strategy="median")),
                        ("escala", StandardScaler())])
    processor = ColumnTransformer([("visuales", numeric, FEATURES)], remainder="drop")
    return Pipeline([("procesamiento", processor), ("clasificador", estimator)])


def evaluate(model, X, y):
    prediction = model.predict(X)
    return {"f1_macro": float(f1_score(y, prediction, labels=CLASSES, average="macro")),
            "reporte": classification_report(y, prediction, labels=CLASSES, output_dict=True, zero_division=0),
            "matriz_confusion": confusion_matrix(y, prediction, labels=CLASSES).tolist()}


def train_models(X, data):
    y = data["clase"]
    train, val, test = [data["split"].eq(s) for s in ["train", "val", "test"]]
    candidates = {"Referencia": DummyClassifier(strategy="most_frequent")}
    for c in [0.1, 1.0, 10.0]:
        candidates[f"Logistica_C{c:g}"] = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)
    for leaf in [2, 5]:
        candidates[f"RandomForest_hoja{leaf}"] = RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_leaf=leaf, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=4)
    results = []
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        started = time.monotonic()
        pipeline.fit(X.loc[train], y.loc[train])
        score = evaluate(pipeline, X.loc[val], y.loc[val])
        results.append({"modelo": name, "f1_macro_val": score["f1_macro"],
                        "recall_blast_val": score["reporte"]["blast"]["recall"],
                        "segundos_ajuste": time.monotonic() - started})
        print(f"Validación {name}: F1 macro={score['f1_macro']:.4f}", flush=True)
    comparison = pd.DataFrame(results).sort_values("f1_macro_val", ascending=False, kind="stable").reset_index(drop=True)
    chosen = comparison.iloc[0]["modelo"]
    # Se cierra la selección usando únicamente validación. Se reentrena con train+val.
    final = make_pipeline(candidates[chosen])
    final.fit(X.loc[train | val], y.loc[train | val])
    final_test = evaluate(final, X.loc[test], y.loc[test])
    baseline = make_pipeline(DummyClassifier(strategy="most_frequent"))
    baseline.fit(X.loc[train | val], y.loc[train | val])
    baseline_test = evaluate(baseline, X.loc[test], y.loc[test])
    return final, comparison, chosen, final_test, baseline_test


def save_bundle(path: Path, pipeline, net, audit, data, comparison, chosen, test_metrics, baseline_metrics):
    bundle = {"formato": "arroz-visual-v1", "features": FEATURES,
              "extractor_state": {k: v.cpu().numpy() for k, v in net.state_dict().items()},
              "pipeline": pipeline, "clases": CLASSES,
              "meta": {"dataset": DATASET, "revision": REVISION, "seed": SEED,
                       "audit": audit, "modelo_elegido": chosen,
                       "validacion": comparison.to_dict("records"), "test": test_metrics,
                       "baseline_test": baseline_metrics,
                       "particiones": pd.crosstab(data["split"], data["clase"]).to_dict(orient="index"),
                       "n_imagenes_sin_copias": len(data), "n_grupos": int(data["grupo"].nunique()),
                       "test_ids": data.loc[data["split"].eq("test"), "id"].tolist(),
                       "versiones": {"torch": torch.__version__, "numpy": np.__version__,
                                     "sklearn": __import__("sklearn").__version__, "pandas": pd.__version__}}}
    joblib.dump(bundle, path, compress=3, protocol=5)
    return bundle


def load_bundle(path=BASE / "model.pkl"):
    bundle = joblib.load(path)
    if bundle.get("formato") != "arroz-visual-v1":
        raise ValueError("Formato de modelo incompatible.")
    net, transform = make_extractor(state=bundle["extractor_state"])
    return bundle, net, transform


def predict_image(image: Image.Image, bundle, net, transform):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    with torch.inference_mode():
        values = net(transform(rgb).unsqueeze(0)).cpu().numpy()
    X = pd.DataFrame(values, columns=bundle["features"])
    probs = bundle["pipeline"].predict_proba(X)[0]
    labels = bundle["pipeline"].named_steps["clasificador"].classes_
    return str(labels[int(np.argmax(probs))]), {str(k): float(v) for k, v in zip(labels, probs)}


def roi_scenarios():
    """Hipótesis de ahorro de tiempo en un año; no son beneficios medidos."""
    rows = []
    for name, cases, minutes in [("Conservador", 500, 1), ("Intermedio", 1500, 3), ("Favorable", 3000, 4)]:
        hour_cost, startup, annual = 50000, 2000000, 500000
        benefit = cases * minutes / 60 * hour_cost
        cost = startup + annual
        rows.append({"escenario": name, "consultas_anuales": cases, "minutos_netos_ahorrados": minutes,
                     "costo_hora_Gs": hour_cost, "inversion_inicial_Gs": startup,
                     "operacion_anual_Gs": annual, "ahorro_anual_Gs": benefit,
                     "beneficio_neto_Gs": benefit - cost, "ROI_primer_anio_pct": 100 * (benefit - cost) / cost})
    return pd.DataFrame(rows)


def run_app():
    import streamlit as st
    st.set_page_config(page_title="Arroz · Asistente visual", page_icon="🌾", layout="centered")
    # El tema nativo mantiene el contraste tanto en modo claro como oscuro.
    st.title("🌾 Asistente visual de arroz")
    st.write("Compará una fotografía con tres categorías del conjunto de referencia.")
    st.caption("Prototipo académico con fotografías de India. Su desempeño en Paraguay aún no fue evaluado.")
    if not (BASE / "model.pkl").exists():
        st.error("Falta model.pkl. Colocá los cinco archivos de la entrega en la misma carpeta.")
        return
    @st.cache_resource
    def resources():
        return load_bundle()
    bundle, net, transform = resources()
    diagnostic, results, about = st.tabs(["Comparar una foto", "Resultados del proyecto", "Cómo funciona"])
    with diagnostic:
        uploaded = st.file_uploader("Fotografía de arroz", type=["jpg", "jpeg", "png", "webp"])
        raw, display_name = None, "fotografia"
        if uploaded is not None:
            raw, display_name = uploaded.getvalue(), uploaded.name
        else:
            sample_path = BASE / "datos_muestra.csv"
            if sample_path.exists():
                @st.cache_data
                def samples():
                    return pd.read_csv(sample_path)
                sample = samples()
                options = ["Elegir ejemplo…"] + [f"{i + 1:02d} · {row.id}" for i, row in enumerate(sample.itertuples())]
                choice = st.selectbox("O usá una fotografía pública de muestra", options)
                if choice != options[0]:
                    idx = options.index(choice) - 1
                    row = sample.iloc[idx]
                    raw, display_name = base64.b64decode(row["imagen_jpeg_base64"]), str(row["id"]) + ".jpg"
        if raw is not None:
            try:
                with Image.open(io.BytesIO(raw)) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    image.load()
                st.image(image, caption=display_name, width=420)
                label, scores = predict_image(image, bundle, net, transform)
                st.subheader(NAMES[label])
                score_table = pd.DataFrame({"Categoría": [NAMES[k] for k in CLASSES],
                                            "Puntaje (%)": [round(scores.get(k, 0) * 100, 1) for k in CLASSES]}).set_index("Categoría")
                st.bar_chart(score_table, color="#79a58a")
                st.caption("Los puntajes comparan las categorías del modelo; no son una certeza diagnóstica. "
                           "Otras enfermedades, otros cultivos y fotos irrelevantes también podrían recibir una categoría.")
                st.download_button("Descargar esta foto", raw, file_name=display_name, mime="image/jpeg")
            except (ValueError, OSError) as exc:
                st.error(f"No se pudo abrir la fotografía: {exc}")
    with results:
        meta, metrics = bundle["meta"], bundle["meta"]["test"]
        col1, col2 = st.columns(2)
        col1.metric("F1 macro · prueba", f"{metrics['f1_macro']:.3f}")
        col2.metric("Recall · piricularia", f"{metrics['reporte']['blast']['recall']:.1%}")
        st.write(f"Evaluación sobre {len(meta['test_ids'])} imágenes reservadas. Modelo: {meta['modelo_elegido']}.")
        st.dataframe(pd.DataFrame(metrics["matriz_confusion"], index=[NAMES[k] for k in CLASSES],
                                  columns=[NAMES[k] for k in CLASSES]), width="stretch")
        st.caption("Filas: etiqueta del conjunto. Columnas: categoría predicha.")
        with st.expander("Tres escenarios económicos hipotéticos"):
            st.write("Estimación del valor del tiempo ahorrado durante un año. Los minutos son netos de revisión profesional. "
                     "Se supone Gs. 50.000/hora, inversión inicial de Gs. 2.000.000 y operación anual de Gs. 500.000.")
            st.dataframe(roi_scenarios().round(1), width="stretch")
            st.caption("Supuestos por validar con un piloto. No se midieron ahorros de insumos ni aumentos de rendimiento.")
    with about:
        st.write("1. Una red visual preentrenada convierte la foto en 576 números.\n\n"
                 "2. Un clasificador aprende a relacionar esos patrones con tres etiquetas.\n\n"
                 "3. Comprobamos sus resultados con fotografías separadas del aprendizaje.")
        st.write("Datos: Paddy Doctor, copia pública de Project-AgML. Etiquetas visuales; no se dispone de diagnósticos de laboratorio.")
        st.markdown("[Ver el conjunto público](https://huggingface.co/datasets/Project-AgML/paddy_disease_classification)")
        st.write("Próximos pasos: validación con imágenes locales, ampliación de enfermedades y evaluación del ahorro de tiempo con técnicos.")


if __name__ == "__main__":
    run_app()
