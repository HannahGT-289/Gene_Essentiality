"""
Fine-tuning of gene essentiality for A. fumigatus A1163 using the pretrained
Ca/Sc/Sp model and three inputs together:
  1) DNA embeddings (CSV under ``input/dna_embeddings/``)
  2) PPI embeddings (CSV under ``input/ppi_embeddings/``)
  3) Ortholog essentiality from Af-to-Ca/Sc/Sp mappings and Ca/Sc/Sp labels
"""
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
TRAIN_DIR = SCRIPT_DIR.parent / "model_pretrain_ca_sc_sp"
TRAIN_INPUT_DIR = TRAIN_DIR / "input"
TRAIN_OUTPUT_DIR = TRAIN_DIR / "output"

# Paths to Af input files
AF_ESS_PATH = INPUT_DIR / "essential_genes" / "Af_essentiality.csv"
AF_DNA_PATH = INPUT_DIR / "dna_embeddings" / "Af_dna_emb.csv"
AF_PPI_FLASHPPI_PATH = INPUT_DIR / "ppi_embeddings" / "Af_ppi_flashppi_emb.csv"

# Paths to Ca/Sc/Sp essentiality CSVs from the pretraining folder
TRAIN_ESS_DIR = TRAIN_INPUT_DIR / "essential_genes"
CA_ESS_PATH = TRAIN_ESS_DIR / "Ca_essentiality.csv"
SC_ESS_PATH = TRAIN_ESS_DIR / "Sc_essentiality.csv"
SP_ESS_PATH = TRAIN_ESS_DIR / "Sp_essentiality.csv"

# Paths to Af ortholog mapping CSVs
ORTHOLOG_DIR = INPUT_DIR / "ortholog_orthofinder"
ORTHOLOG_PATHS = {
    ("Af", "Ca"): ORTHOLOG_DIR / "Af_ortholog_in_Ca.csv",
    ("Af", "Sc"): ORTHOLOG_DIR / "Af_ortholog_in_Sc.csv",
    ("Af", "Sp"): ORTHOLOG_DIR / "Af_ortholog_in_Sp.csv",
}

CHECKPOINT_BASENAME_FULL = "joint_three_species_dna_flashppi_orth3_masked.full.pt"
CHECKPOINT_BASENAME_PTH = "joint_three_species_dna_flashppi_orth3_masked.pth"

# Hyperparameters
SPECIES = ["Ca", "Sc", "Sp"]
DNA_DIM = 4096
PPI_DIM = 128
HIDDEN_DIM = 256
HIDDEN_DIM2 = 16
ORTH_DIM = 3
LEARNING_RATE = 4e-5
WEIGHT_DECAY = 0.05
DROPOUT_RATE = 0.3
BATCH_SIZE = 256
EPOCHS = 500
PATIENCE = 20
VAL_FRACTION = 0.2
SEED = 42
DNA_FEATURE_PREFIX = "x"
PPI_FEATURE_PREFIX = "x"


def resolve_checkpoint_path() -> Path:
    """Resolve the pretrained Ca/Sc/Sp checkpoint from the pretraining folder."""
    candidates = [
        TRAIN_OUTPUT_DIR / CHECKPOINT_BASENAME_FULL,
        TRAIN_OUTPUT_DIR / CHECKPOINT_BASENAME_PTH,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find joint checkpoint. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def _get_device() -> str:
    """Use CUDA when available; otherwise use Apple Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _get_device()


def _effective_batch_size(n: int) -> int:
    """Keep the final batch size valid when a split has fewer rows than BATCH_SIZE."""
    return max(1, min(BATCH_SIZE, n))


def load_dna_embeddings_as_dict(path: Path, x_prefix: str = DNA_FEATURE_PREFIX):
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    x_cols = [c for c in df.columns if c.startswith(x_prefix) and c[len(x_prefix) :].replace("_", "").isdigit()]
    if not x_cols:
        x_cols = [c for c in df.columns if c != "Feature_Name"]
    if not x_cols or not all(c.startswith(x_prefix) for c in x_cols):
        raise ValueError(f"DNA embedding columns must start with '{x_prefix}' in {path}")
    out = {row["Feature_Name"]: row[x_cols].values.astype(np.float32) for _, row in df.iterrows()}
    return out, x_cols


def load_ppi_embeddings(ppi_path: Path, feature_name_col: str = "Feature_Name"):
    df = pd.read_csv(ppi_path)
    if df.columns[0] != feature_name_col and "gene_id" in df.columns:
        df = df.rename(columns={"gene_id": feature_name_col})
    elif df.columns[0] != feature_name_col:
        df = df.rename(columns={df.columns[0]: feature_name_col})
    feat_cols = [
        c
        for c in df.columns
        if c.startswith(PPI_FEATURE_PREFIX) and c[len(PPI_FEATURE_PREFIX) :].replace("_", "").isdigit()
    ]
    if not feat_cols:
        feat_cols = [c for c in df.columns if c != feature_name_col]
    if not feat_cols or not all(c.startswith(PPI_FEATURE_PREFIX) for c in feat_cols):
        raise ValueError(f"PPI columns must start with '{PPI_FEATURE_PREFIX}' in {ppi_path}")
    return df[[feature_name_col] + feat_cols], feat_cols


class fungal_gene_ess_dataset(Dataset):
    def __init__(self, df, dna_cols, ppi_cols, orth_by_gene):
        self.df = df.reset_index(drop=True)
        self.dna_cols = dna_cols
        self.ppi_cols = ppi_cols
        self.orth_by_gene = orth_by_gene

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        g = row["Feature_Name"]
        dna = row[self.dna_cols].values.astype(np.float32)
        orth = self.orth_by_gene[g]
        ppi = row[self.ppi_cols].values.astype(np.float32)
        y = np.float32(row["y"])
        return dna, orth, ppi, y


def collate_batch_ppi(batch):
    dna_list, orth_list, ppi_list, y_list = zip(*batch)
    dna = torch.tensor(np.stack(dna_list), dtype=torch.float32)
    orth = torch.tensor(np.stack(orth_list), dtype=torch.float32)
    ppi = torch.tensor(np.stack(ppi_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32).unsqueeze(1)
    return dna, orth, ppi, y


class multimodal_bilinear(nn.Module):
    """DNA + ortholog + PPI with three bilinear synergy terms."""

    def __init__(self, dna_dim, ppi_dim, orth_dim, hidden_dim, hidden_dim2, dropout_rate):
        super().__init__()
        self.dna_encoder = nn.Sequential(
            nn.LayerNorm(dna_dim),
            nn.Linear(dna_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
        )
        self.ortholog_encoder = nn.Linear(orth_dim, hidden_dim2)
        self.ppi_encoder = nn.Sequential(
            nn.LayerNorm(ppi_dim),
            nn.Linear(ppi_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
        )
        self.synergy_ppi_dna = nn.Bilinear(hidden_dim2, hidden_dim2, hidden_dim2)
        self.synergy_ppi_ortholog = nn.Bilinear(hidden_dim2, hidden_dim2, hidden_dim2)
        self.synergy_dna_ortholog = nn.Bilinear(hidden_dim2, hidden_dim2, hidden_dim2)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim2 * 6, hidden_dim2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x_dna, x_orth, x_ppi):
        z_dna = self.dna_encoder(x_dna)
        z_ortholog = self.ortholog_encoder(x_orth)
        z_ppi = self.ppi_encoder(x_ppi)
        z_ppi_dna = torch.tanh(self.synergy_ppi_dna(z_ppi, z_dna))
        z_ppi_ortholog = torch.tanh(self.synergy_ppi_ortholog(z_ppi, z_ortholog))
        z_dna_ortholog = torch.tanh(self.synergy_dna_ortholog(z_dna, z_ortholog))
        x = torch.cat((z_dna, z_ortholog, z_ppi, z_ppi_dna, z_ppi_ortholog, z_dna_ortholog), dim=1)
        return self.output_head(x)


def load_ortholog_mapping(path: Path, orth_col: str = "Orthologs_Feature") -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    if not path.exists():
        return dict(out)
    df = pd.read_csv(path)
    if orth_col not in df.columns:
        orth_col = df.columns[1]
    for _, row in df.iterrows():
        out[str(row["Feature_Name"]).strip()].append(str(row[orth_col]).strip())
    return dict(out)


def load_essentiality_simple(path):
    df = pd.read_csv(path)
    if "Essentiality" not in df.columns or "Feature_Name" not in df.columns:
        raise ValueError(f"Missing columns in {path}")
    df["Feature_Name"] = df["Feature_Name"].astype(str).str.strip()
    if df.duplicated(subset=["Feature_Name"]).any():
        dupes = df[df.duplicated(subset=["Feature_Name"], keep=False)]["Feature_Name"].unique().tolist()
        raise ValueError(f"Feature_Name must be unique in {path}. Duplicates: {dupes[:10]}{'...' if len(dupes) > 10 else ''}")
    return dict(zip(df["Feature_Name"], df["Essentiality"]))


def _compute_ortholog_ess(
    gene: str,
    target_sp: str,
    ortholog_maps: dict,
    ess_dicts: dict[str, dict],
) -> float:
    """
    Collapse Af ortholog labels against one training species.

    Any essential mapped ortholog gives 1. If the mapped binary labels are only
    nonessential, use -1. Missing orthologs and non-binary labels remain 0.
    """
    path = ORTHOLOG_PATHS.get(("Af", target_sp))
    if not path or not path.exists():
        return 0.0
    key = ("Af", target_sp)
    orth_map = ortholog_maps.get(key)
    if orth_map is None:
        orth_map = load_ortholog_mapping(path)
        ortholog_maps[key] = orth_map
    ess_dict = ess_dicts[target_sp]
    vals = [
        ess_dict.get(orth_id)
        for orth_id in orth_map.get(str(gene).strip(), [])
        if ess_dict.get(orth_id) in (1, -1)
    ]
    if 1 in vals:
        return 1.0
    if -1 in vals:
        return -1.0
    return 0.0


def build_af_ortholog(
    genes: list[str],
    ess_dicts: dict[str, dict],
    ortholog_maps: dict,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for g in genes:
        dims = [float(_compute_ortholog_ess(g, sp, ortholog_maps, ess_dicts)) for sp in SPECIES]
        out[g] = np.array(dims, dtype=np.float32)
    return out


def load_training_essentiality():
    ca_ess = load_essentiality_simple(CA_ESS_PATH)
    sc_ess = load_essentiality_simple(SC_ESS_PATH)
    sp_ess = load_essentiality_simple(SP_ESS_PATH)
    return {"Ca": ca_ess, "Sc": sc_ess, "Sp": sp_ess}


def load_af_essentiality(label_path: Path) -> pd.DataFrame:
    if label_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(label_path, sheet_name="A. fumigatus A1163")
    else:
        df = pd.read_csv(label_path)
    if "Feature_Name" not in df.columns or "Essentiality" not in df.columns:
        raise ValueError(f"Expected columns Feature_Name, Essentiality in {label_path}, got {list(df.columns)}")
    df = df[["Feature_Name", "Essentiality"]].copy()
    df["Feature_Name"] = df["Feature_Name"].astype(str).str.strip()
    df = df[df["Essentiality"].isin([1, -1])].copy()
    df["y"] = df["Essentiality"].apply(lambda v: 1.0 if v == 1 else 0.0)
    return df.reset_index(drop=True)


def build_all_feature_table(
    dna_path: Path,
    ppi_path: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    emb_dict, x_cols = load_dna_embeddings_as_dict(dna_path)
    dna_cols = [f"dna_{i}" for i in range(len(x_cols))]
    if len(dna_cols) != DNA_DIM:
        raise ValueError(f"DNA dim {len(dna_cols)} != DNA_DIM ({DNA_DIM})")

    ppi_cols = [f"ppi_{i}" for i in range(PPI_DIM)]

    if ppi_path.exists():
        ppi_df, ppi_raw_cols = load_ppi_embeddings(ppi_path)
        if len(ppi_raw_cols) != PPI_DIM:
            raise ValueError(f"PPI dim {len(ppi_raw_cols)} != PPI_DIM ({PPI_DIM})")
        rename_ppi = dict(zip(ppi_raw_cols, ppi_cols))
        ppi_df = ppi_df.rename(columns=rename_ppi)
        ppi_df = ppi_df[["Feature_Name"] + ppi_cols]
        dup_mask = ppi_df["Feature_Name"].duplicated(keep=False)
        if dup_mask.any():
            dups = ppi_df.loc[dup_mask, "Feature_Name"].unique().tolist()
            preview = dups[:20]
            more = f" ... and {len(dups) - len(preview)} more" if len(dups) > len(preview) else ""
            raise ValueError(
                f"{ppi_path.name}: expected one row per Feature_Name; found duplicates for "
                f"{len(dups)} gene(s), e.g. {preview}{more}"
            )
    else:
        ppi_df = None

    rows = []
    for g, vec in emb_dict.items():
        row = {"Feature_Name": g}
        for i, c in enumerate(dna_cols):
            row[c] = vec[i]
        rows.append(row)

    base = pd.DataFrame(rows)
    if base.empty:
        raise RuntimeError("No Af genes were found in the DNA embeddings.")

    if ppi_df is None:
        for c in ppi_cols:
            base[c] = 0.0
    else:
        base = base.merge(ppi_df, on="Feature_Name", how="left")
        base[ppi_cols] = base[ppi_cols].fillna(0.0)

    return base, dna_cols, ppi_cols


def build_labeled_feature_table(all_feature_df: pd.DataFrame, lab_df: pd.DataFrame) -> pd.DataFrame:
    labeled_cols = lab_df[["Feature_Name", "Essentiality", "y"]].copy()
    out = all_feature_df.merge(labeled_cols, on="Feature_Name", how="inner")
    if out.empty:
        raise RuntimeError("No overlap between Af labels and DNA embeddings.")
    return out


def load_pretrained_state(ckpt_path: Path, device: str) -> tuple[dict, int]:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state = raw["state_dict"]
        orth_dim = int(raw.get("orth_dim", ORTH_DIM))
    else:
        state = raw
        orth_dim = ORTH_DIM
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    if state and all(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state, orth_dim


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    df: pd.DataFrame,
    dna_cols: list[str],
    ppi_cols: list[str],
    orth_by_gene: dict[str, np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    probs = []
    n = len(df)
    for start in range(0, n, batch_size):
        batch = df.iloc[start : start + batch_size]
        dna = torch.tensor(batch[dna_cols].values, dtype=torch.float32, device=device)
        ppi = torch.tensor(batch[ppi_cols].values, dtype=torch.float32, device=device)
        orth = torch.tensor(
            np.stack([orth_by_gene[g] for g in batch["Feature_Name"].values]),
            dtype=torch.float32,
            device=device,
        )
        logits = model(dna, orth, ppi)
        probs.append(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.concatenate(probs)


def train_af_a1163(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dna_cols: list[str],
    ppi_cols: list[str],
    orth_by_gene: dict,
    orth_dim: int,
    pretrained_state: dict,
    init_checkpoint_path: Path,
    out_pth: Path,
    device: str,
) -> tuple[float, float, dict, np.ndarray, nn.Module]:
    train_ds = fungal_gene_ess_dataset(train_df, dna_cols, ppi_cols, orth_by_gene)
    val_ds = fungal_gene_ess_dataset(val_df, dna_cols, ppi_cols, orth_by_gene)
    train_bs = _effective_batch_size(len(train_ds))
    val_bs = _effective_batch_size(len(val_ds))
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        collate_fn=collate_batch_ppi,
        generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, collate_fn=collate_batch_ppi)

    print(f"  DataLoader batch sizes: train={train_bs}  val={val_bs}  (batch_size={BATCH_SIZE})")

    model = multimodal_bilinear(
        DNA_DIM,
        PPI_DIM,
        orth_dim,
        HIDDEN_DIM,
        HIDDEN_DIM2,
        DROPOUT_RATE,
    ).to(device)
    model.load_state_dict(pretrained_state, strict=True)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    last_train_loss = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            dna, orth, ppi, y = batch
            dna, orth, ppi, y = dna.to(device), orth.to(device), ppi.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(dna, orth, ppi)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                dna, orth, ppi, y = batch
                dna, orth, ppi, y = dna.to(device), orth.to(device), ppi.to(device), y.to(device)
                val_loss += criterion(model(dna, orth, ppi), y).item() * len(y)
        v_loss = val_loss / max(1, len(val_ds))
        print(f"  Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_pth)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch + 1} (patience={PATIENCE})")
                break
        last_train_loss = avg_train_loss

    if best_state is None:
        best_state = model.state_dict()
        torch.save(best_state, out_pth)
    else:
        model.load_state_dict(best_state)
    model.eval()

    infer_bs = _effective_batch_size(len(val_df))
    val_probs = predict_probs(model, val_df, dna_cols, ppi_cols, orth_by_gene, device, infer_bs)
    metrics: dict = {
        "Val_Loss_best": best_val_loss,
    }
    val_y = val_df["y"].values.astype(np.float32)
    if len(np.unique(val_y)) >= 2:
        metrics["AUROC"] = float(roc_auc_score(val_y, val_probs))
        metrics["AUPRC"] = float(average_precision_score(val_y, val_probs))
    else:
        metrics["AUROC"] = float("nan")
        metrics["AUPRC"] = float("nan")

    torch.save(
        {
            "state_dict": best_state,
            "orth_dim": orth_dim,
            "dna_dim": DNA_DIM,
            "ppi_dim": PPI_DIM,
            "hidden_dim": HIDDEN_DIM,
            "hidden_dim2": HIDDEN_DIM2,
            "species": "Af",
            "init_checkpoint": str(init_checkpoint_path.resolve()),
        },
        out_pth.with_suffix(".full.pt"),
    )

    return last_train_loss, best_val_loss, metrics, val_probs, model


def run_finetune_af_a1163():
    device = DEVICE

    ckpt_path = resolve_checkpoint_path()
    ckpt_resolved = ckpt_path.resolve()
    print("Init checkpoint:", ckpt_resolved)

    pretrained_state, orth_dim_ckpt = load_pretrained_state(ckpt_path, device)
    print(f"  orth_dim: {orth_dim_ckpt}")

    if not AF_DNA_PATH.exists():
        raise FileNotFoundError(f"DNA embeddings not found: {AF_DNA_PATH}")

    print("Loading labels:", AF_ESS_PATH)
    lab_df = load_af_essentiality(AF_ESS_PATH)
    print(f"  Labeled genes (Essentiality in {{1,-1}}): {len(lab_df)}")

    print("Loading Ca/Sc/Sp essentiality from:", TRAIN_ESS_DIR)
    ess_dicts = load_training_essentiality()

    print("Building features (DNA + PPI + ortholog essentiality)...")
    all_feat_df, dna_cols, ppi_cols = build_all_feature_table(AF_DNA_PATH, AF_PPI_FLASHPPI_PATH)
    feat_df = build_labeled_feature_table(all_feat_df, lab_df)
    if feat_df.empty:
        raise RuntimeError("No genes with both label and DNA embedding; cannot train.")

    ortholog_maps: dict = {}
    orth_by_gene = build_af_ortholog(feat_df["Feature_Name"].tolist(), ess_dicts, ortholog_maps)
    all_orth_by_gene = build_af_ortholog(all_feat_df["Feature_Name"].tolist(), ess_dicts, ortholog_maps)

    n = len(feat_df)
    if feat_df["y"].nunique() < 2:
        raise RuntimeError("Need both classes in genes with DNA embeddings.")

    stratify = feat_df["y"] if feat_df["y"].nunique() > 1 and n >= 10 else None
    try:
        train_df, val_df = train_test_split(
            feat_df,
            test_size=VAL_FRACTION,
            stratify=stratify,
            random_state=SEED,
        )
    except ValueError:
        train_df, val_df = train_test_split(
            feat_df, test_size=VAL_FRACTION, stratify=None, random_state=SEED
        )

    print(
        f"Train / validation: n_train={len(train_df)}  n_val={len(val_df)}  "
        f"(val_fraction={VAL_FRACTION}, batch_size={BATCH_SIZE})"
    )

    out_pth = OUTPUT_DIR / "afumigatus_from_joint_best_val.pth"

    tr_loss, val_loss, val_metrics, _val_probs, model = train_af_a1163(
        train_df=train_df,
        val_df=val_df,
        dna_cols=dna_cols,
        ppi_cols=ppi_cols,
        orth_by_gene=orth_by_gene,
        orth_dim=orth_dim_ckpt,
        pretrained_state=pretrained_state,
        init_checkpoint_path=ckpt_resolved,
        out_pth=out_pth,
        device=device,
    )

    all_probs = predict_probs(
        model,
        all_feat_df,
        dna_cols,
        ppi_cols,
        all_orth_by_gene,
        device,
        _effective_batch_size(len(all_feat_df)),
    )
    train_genes = set(train_df["Feature_Name"].tolist())
    val_genes = set(val_df["Feature_Name"].tolist())
    all_out = all_feat_df[["Feature_Name"]].copy()
    all_out["y_prob"] = all_probs
    all_out["used_in_finetuning"] = ""
    all_out.loc[all_out["Feature_Name"].isin(train_genes), "used_in_finetuning"] = "Train"
    all_out.loc[all_out["Feature_Name"].isin(val_genes), "used_in_finetuning"] = "Val"
    output_labels = lab_df[["Feature_Name", "Essentiality"]].copy()
    all_out = all_out.merge(output_labels, on="Feature_Name", how="left")
    all_out["Essentiality"] = all_out["Essentiality"].astype("Int64")
    all_csv = OUTPUT_DIR / "afumigatus_all_gene_probabilities.csv"
    all_out.to_csv(all_csv, index=False)

    summary = {
        "n_labeled_af_genes": len(lab_df),
        "n_all_af_genes_with_dna": len(all_feat_df),
        "n_labeled_with_dna": n,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "Train_Loss_last": tr_loss,
        "init_checkpoint": str(ckpt_resolved),
        "out_pth": str(out_pth.resolve()),
        "out_full_pt": str(out_pth.with_suffix(".full.pt").resolve()),
        "all_gene_probabilities_csv": str(all_csv.resolve()),
        "seed": SEED,
        "lr": LEARNING_RATE,
        **{k: val_metrics[k] for k in val_metrics},
    }
    summary_path = OUTPUT_DIR / "afumigatus_train_val_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n" + "-" * 60)
    print("  Validation set")
    print(f"  Val loss (best): {val_loss:.4f}")
    for k in ("AUROC", "AUPRC"):
        if k in val_metrics:
            print(f"  {k}: {val_metrics[k]:.4f}" if isinstance(val_metrics[k], float) else f"  {k}: {val_metrics[k]}")
    print(f"\n  Weights:     {out_pth}")
    print(f"  Full ckpt:   {out_pth.with_suffix('.full.pt')}")
    print(f"  All probs:   {all_csv}")
    print(f"  Summary:     {summary_path}")
    print("-" * 60)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print("Output directory:", OUTPUT_DIR)
    run_finetune_af_a1163()


if __name__ == "__main__":
    main()
