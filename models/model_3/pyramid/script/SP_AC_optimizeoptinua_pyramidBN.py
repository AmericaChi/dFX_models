# %%
# ================================================================
# 📦 Setup: Libraries, Config, Utilities (for Optuna + NN training)
# ================================================================

# --- Basic Python tools ---
import os, json, math, random, datetime, uuid, warnings
warnings.filterwarnings("ignore")

# --- Data and math handling ---
import numpy as np
import pandas as pd
from pathlib import Path

# --- PyTorch for building NN ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# --- Save and plot tools ---
import joblib
import matplotlib.pyplot as plt

# --- ML helpers ---
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# --- Optuna for hyperparameter optimization ---
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import NopPruner, SuccessiveHalvingPruner

# ================================================================
# 🔁 Reproducibility (fix randomness so results are same each run)
# ================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# ⚙️ Device Selection (GPU / MPS / CPU)
# ================================================================
# "mps" -> Apple Silicon GPU (M1/M2/M3)
# "cuda" -> NVIDIA GPU
# "cpu" -> fallback if no GPU
DEVICE = (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print("Device:", DEVICE)

# ================================================================
# 📁 Create unique output folder for results
# ================================================================
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# Use current time + random ID for run name
RUN_ID  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
OUT_DIR = RUNS_DIR / RUN_ID

# Make subfolders for storing outputs
for sub in ["cv", "final", "models", "plots", "external"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

print("Results will be saved in:", OUT_DIR.resolve())

# ================================================================
# ⚙️ Global Training Settings
# ================================================================
SEARCH_TRIALS   = 500      # How many Optuna trials to try
SEARCH_FOLDS    = 10         # Cross-validation folds
PATIENCE_SEARCH = 12        # Early stop patience during search
EPOCHS_SEARCH   = 100       # Max epochs during search

PATIENCE_FINAL  = 80        # Early stop patience for final model
EPOCHS_FINAL    = 400       # Max epochs for final training

# ================================================================
# 🛠️ Utility Functions
# ================================================================

# Save Python object as JSON file (useful for saving results)
def log_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))

# Create bins for continuous targets (for stratified CV in regression)
def make_bins_for_stratify(y, n_bins=10):
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.unique(np.quantile(y, qs))
    if len(edges) <= 2:
        edges = np.linspace(y.min(), y.max(), n_bins+1)
    return np.digitize(y, edges[1:-1], right=True)

# Calculate all important regression metrics
def metrics_full(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred) if len(np.unique(y_true)) > 1 else (np.nan, None)
    sr, _ = spearmanr(y_true, y_pred) if len(np.unique(y_true)) > 1 else (np.nan, None)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson": float(pr),
        "Spearman": float(sr)
    }

# Scatter plot of predicted vs actual with metrics annotated
def plot_scatter_corr(y_true, y_pred, title, outpath, metrics=None, show_identity=True):
    m = metrics if metrics is not None else metrics_full(y_true, y_pred)

    plt.figure(figsize=(5,5))
    ax = plt.gca()

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.2 * (hi - lo if hi > lo else 1.0)
    xlim = (lo - pad, hi + pad)
    ylim = (lo - pad, hi + pad)

    ax.scatter(y_true, y_pred, alpha=0.6)

    # Draw y=x line (ideal prediction)
    if show_identity:
        ax.plot(xlim, ylim)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)

    # Annotate metrics on the plot
    txt = (
        f"MAE = {m['MAE']:.4f}\n"
        f"RMSE = {m['RMSE']:.4f}\n"
        f"R² = {m['R2']:.4f}\n"
        f"r = {m['Pearson']:.4f}\n"
        f"ρ = {m['Spearman']:.4f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# Residuals histogram (y_true - y_pred)
def plot_residuals(y_true, y_pred, title, outpath):
    res = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=30)
    plt.title(title)
    plt.xlabel("Residuals (y_true - y_pred)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# %%
# ================================================================
# 📊 Data Loading, Feature Selection & Train/Test Split (80/20)
# ================================================================

# 📁 Path to your dataset (tab-separated file)
DATA_CSV   = Path("../../datasets/binding/doubles.txt")
TARGET_COL = "diff"      # The column we want to predict (target variable)
N_FEATURES = 13          # Number of features to use in training

# ------------------------------------------------
# 🧠 Step 1: Load the dataset
# ------------------------------------------------
df = pd.read_csv(DATA_CSV, sep="\t")  # Read as a DataFrame
assert TARGET_COL in df.columns, f"❌ Target column '{TARGET_COL}' not found!"

# ------------------------------------------------
# 🧹 Step 2: Drop unnecessary columns
# ------------------------------------------------
# These are metadata or irrelevant columns for model training
drop_cols = [
    "system", "mutation", "loop_entropy", "disulfide", "partial covalent interactions",
    "Entropy Complex", "Energy_SolvP", "Energy_SolvH", "cis_bond"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# ------------------------------------------------
# 🔍 Step 3: Check if special columns (FoldX or experimental ΔΔG) exist
# ------------------------------------------------
has_foldx = "ddG_foldx" in df.columns
has_exp   = "exp_ddG"   in df.columns

# If present, store them as numpy arrays for later analysis
fx_all = df["ddG_foldx"].to_numpy(np.float32) if has_foldx else None
ex_all = df["exp_ddG"].to_numpy(np.float32)   if has_exp else None

# ------------------------------------------------
# 🧪 Step 4: Select feature columns for model training
# ------------------------------------------------
# Remove target and extra columns, then pick the first N features
feature_cols = [c for c in df.columns if c not in [TARGET_COL, "ddG_foldx", "exp_ddG"]][:N_FEATURES]
assert len(feature_cols) >= N_FEATURES, f"❌ Expected ≥{N_FEATURES} features, found {len(feature_cols)}"

# Convert to numpy arrays for ML
X_all = df[feature_cols].to_numpy(np.float32)  # input features
y_all = df[TARGET_COL].to_numpy(np.float32)    # target values

# ------------------------------------------------
# 📦 Step 5: Stratify target for regression
# ------------------------------------------------
# Because y is continuous, we create bins to help keep distribution balanced in train/test
bins = make_bins_for_stratify(y_all, n_bins=10)

# ------------------------------------------------
# ✂️ Step 6: Split data into 80% train / 20% test
# ------------------------------------------------
# Depending on which optional columns exist, we split them too
if has_foldx and has_exp:
    X_tr, X_te, y_tr, y_te, fx_tr, fx_te, ex_tr, ex_te = train_test_split(
        X_all, y_all, fx_all, ex_all,
        test_size=0.2, random_state=SEED, stratify=bins
    )
elif has_foldx:
    X_tr, X_te, y_tr, y_te, fx_tr, fx_te = train_test_split(
        X_all, y_all, fx_all,
        test_size=0.2, random_state=SEED, stratify=bins
    )
    ex_tr = ex_te = None
elif has_exp:
    X_tr, X_te, y_tr, y_te, ex_tr, ex_te = train_test_split(
        X_all, y_all, ex_all,
        test_size=0.2, random_state=SEED, stratify=bins
    )
    fx_tr = fx_te = None
else:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=0.2, random_state=SEED, stratify=bins
    )
    fx_tr = fx_te = ex_tr = ex_te = None

# ------------------------------------------------
# 💾 Step 7: Save raw arrays (for reproducibility or future use)
# ------------------------------------------------
np.save(OUT_DIR / "X_train_raw.npy", X_tr)
np.save(OUT_DIR / "y_train.npy",    y_tr)
np.save(OUT_DIR / "X_test_raw.npy", X_te)
np.save(OUT_DIR / "y_test.npy",     y_te)

if fx_te is not None:
    np.save(OUT_DIR / "ddG_foldx_test.npy", fx_te)
if ex_te is not None:
    np.save(OUT_DIR / "exp_ddG_test.npy",   ex_te)

# ------------------------------------------------
# 📜 Step 8: Save metadata about this run
# ------------------------------------------------
log_json({
    "run_id": Path(OUT_DIR).name,
    "seed": SEED,
    "feature_cols": feature_cols,
    "target_col": TARGET_COL,
    "n_train": int(len(X_tr)),
    "n_test": int(len(X_te)),
    "has_foldx": bool(has_foldx),
    "has_exp": bool(has_exp)
}, OUT_DIR / "run_info.json")

# ✅ Final check
print("Train/Test split shapes:", X_tr.shape, X_te.shape)


# %%
# ================================================================
# 🧠 MLP Model + One-Fold Training Function
# ================================================================

# ------------------------------------------------
# 🧱 1. Define a simple MLP model
# ------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, layers=3, width=256, dropout=0.1, activation="relu", use_bn=True):
        super().__init__()
        Act = nn.ReLU if activation == "relu" else nn.GELU

        seq = []
        d = in_dim

        # 📉 Pyramid structure: layer width halves each time
        widths = [max(8, width // (2 ** i)) for i in range(layers)]

        for w in widths:
            seq += [nn.Linear(d, w)]
            if use_bn:
                seq += [nn.BatchNorm1d(w)]  # ✅ BatchNorm added
            seq += [Act()]
            seq += [nn.Dropout(dropout)]
            d = w

        # Final layer (regression output)
        seq += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------------------------------
# ⚖️ 2. Choose the loss function
# ------------------------------------------------
def pick_loss(name):
    name = name.lower()
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    elif name == "mse":
        return nn.MSELoss()
    else:
        return nn.L1Loss()  # MAE

# ------------------------------------------------
# 🔁 3. Train the model for one fold
# ------------------------------------------------
def train_one_fold(Xtr, ytr, Xva, yva, cfg, device=DEVICE, cv_mode=True, save_prefix=None):
    """
    Trains one fold of the MLP model.
    - Xtr, ytr: training data
    - Xva, yva: validation data
    - cfg: hyperparameters (dict)
    - cv_mode=True: disables saving plots/models during CV
    """

    # ✅ Step 1: Scale features (important for NN training)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)

    # ✅ Step 2: Create DataLoaders (mini-batch loaders)
    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr)),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xva_s), torch.from_numpy(yva)),
        batch_size=cfg["batch_size"], shuffle=False
    )

    # ✅ Step 3: Initialize model, optimizer, loss, and scheduler
    model = MLP(
        in_dim=Xtr.shape[1],
        layers=cfg["layers"],
        width=cfg["width"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],
        use_bn=True
    ).to(device)

    opt   = AdamW(model.parameters(), lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
    #opt   = torch.optim.Adam(model.parameters(), lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
    lossf = pick_loss(cfg["loss"])

    steps_per_epoch = max(1, math.ceil(len(Xtr_s) / cfg["batch_size"]))
    sched = OneCycleLR(
        opt,
        max_lr=cfg["max_lr"],
        epochs=cfg["epochs"],
        steps_per_epoch=steps_per_epoch,
        pct_start=cfg["pct_start"],
        final_div_factor=1e5
    )

    # ✅ Step 4: Train loop setup
    best = float("inf")      # best validation loss
    best_state = None        # save best model weights
    bad = 0                  # counter for early stopping
    tr_losses = []           # training loss history
    va_losses = []           # validation loss history

    # ------------------------------------------------
    # 🚀 Training Loop (epoch by epoch)
    # ------------------------------------------------
    for ep in range(cfg["epochs"]):
        model.train()
        tl = 0.0  # training loss

        # 🔁 Train step: forward → loss → backward → update
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward()

            # Prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            sched.step()
            tl += loss.item() * len(xb)
        tl /= len(Xtr_s)

        # 🔍 Validation step
        model.eval()
        vl = 0.0
        preds = []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                pr = model(xb)
                vl += lossf(pr, yb).item() * len(xb)
                preds.append(pr.detach().cpu().numpy())
        vl /= len(Xva_s)
        preds = np.concatenate(preds)

        # Save losses for plotting
        tr_losses.append(tl)
        va_losses.append(vl)

        # ✅ Early stopping logic
        if vl + 1e-9 < best:
            best = vl
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if bad >= cfg["patience"]:  # Stop if no improvement for N epochs
            break

    # ✅ Load best model weights
    model.load_state_dict(best_state)

    # ------------------------------------------------
    # 📊 Step 5: Compute metrics on validation set
    # ------------------------------------------------
    with torch.no_grad():
        y_pred = model(torch.from_numpy(Xva_s).to(device)).detach().cpu().numpy()
    m = metrics_full(yva, y_pred)

    # ------------------------------------------------
    # 📈 Step 6: Save plots (only in final training, not during CV)
    # ------------------------------------------------
    if not cv_mode and save_prefix is not None:
        # Loss curve
        plt.figure(figsize=(6, 4))
        plt.plot(tr_losses, label="train")
        plt.plot(va_losses, label="val")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_loss.png", dpi=200)
        plt.close()

        # Scatter and residual plots
        plot_scatter_corr(yva, y_pred, "Val: true vs pred", f"{save_prefix}_scatter.png", metrics=m)
        plot_residuals(yva, y_pred, "Val residuals", f"{save_prefix}_residuals.png")

    return model, scaler, m


# %%
# ================================================================
# 🔍 Hyperparameter Search: Optuna + ASHA + 5-fold CV
# (No models/plots are saved during CV — only metrics)
# ================================================================

# ------------------------------------------------
# 🎲 1. Define how each trial's hyperparameters are sampled
# ------------------------------------------------
def sample_cfg_from_trial(trial):
    # Ask Optuna to choose hyperparameters for this trial
    layers   = trial.suggest_categorical("layers", [2, 3, 4, 5, 6])       # number of hidden layers
    width    = trial.suggest_categorical("width",  [128, 256, 512, 1024])       # number of neurons per layer
    dropout  = trial.suggest_categorical("dropout",[0.0, 0.1, 0.2, 0.3])     # dropout rate
    batch    = trial.suggest_categorical("batch_size", [16, 32, 64])         # batch size
    loss     = trial.suggest_categorical("loss", ["huber", "mse", "mae"])    # loss function
    act      = trial.suggest_categorical("activation", ["relu", "gelu"])     # activation function
    max_lr   = trial.suggest_float("max_lr", 5e-4, 2e-3, log=True)           # max learning rate
    wd       = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)     # weight decay (L2 regularization)
    pct      = trial.suggest_float("pct_start", 0.2, 0.4)                    # OneCycleLR warmup pct

    # 🛑 Optional rule: if network is too wide (256), enforce some dropout
    if width == 256 and dropout < 0.1:
        dropout = 0.1
        trial.set_user_attr("constraint_adjusted_dropout", True)

    # Return configuration as a dictionary
    return dict(
        layers=layers, width=width, dropout=dropout, batch_size=batch,
        loss=loss, activation=act, max_lr=max_lr, weight_decay=wd,
        pct_start=pct, epochs=EPOCHS_SEARCH, patience=PATIENCE_SEARCH
    )

# ------------------------------------------------
# 📊 2. Define the objective function for Optuna (5-fold CV)
# ------------------------------------------------
def search_objective_5fold(X, y):
    # Create bins for stratified CV (for regression)
    y_bins = make_bins_for_stratify(y, n_bins=10)
    skf = StratifiedKFold(n_splits=SEARCH_FOLDS, shuffle=True, random_state=SEED)

    def _obj(trial):
        # Sample hyperparameters for this trial
        cfg = sample_cfg_from_trial(trial)
        fold_mae = []

        # 🔁 Train & validate model for each fold
        for k, (tr_idx, va_idx) in enumerate(skf.split(X, y_bins), start=1):
            _, _, m = train_one_fold(
                X[tr_idx], y[tr_idx],
                X[va_idx], y[va_idx],
                cfg, device=DEVICE,
                cv_mode=True,      # ⚠️ do not save models/plots during CV
                save_prefix=None
            )
            fold_mae.append(m["MAE"])     # collect MAE for this fold

            # Report intermediate result to Optuna (for pruning)
            trial.report(m["MAE"], step=k)
            if trial.should_prune():      # stop trial early if it's performing poorly
                raise optuna.TrialPruned()

        # 📉 Calculate final mean MAE across folds
        # (Optional) small penalty if network is large but not efficient
        score = float(np.mean(fold_mae)) + (0.01 if (cfg["layers"] == 3 and cfg["width"] > 128) else 0.0)
        return score

    return _obj

# ------------------------------------------------
# 🚀 3. Create and run the Optuna study
# ------------------------------------------------
study = optuna.create_study(
    direction="minimize",           # we want to minimize MAE
    sampler=TPESampler(seed=SEED),  # TPE = Bayesian sampler (smarter than random)
    pruner=NopPruner()              # no early pruning (can use ASHA here)
)

# Run the search with 5-fold CV objective
study.optimize(search_objective_5fold(X_tr, y_tr), n_trials=SEARCH_TRIALS, gc_after_trial=True)

# ------------------------------------------------
# 📁 4. Save and report results
# ------------------------------------------------
# Save all trial results to CSV
df_trials = study.trials_dataframe()
df_trials.to_csv(OUT_DIR / "cv" / "hpo_summary.csv", index=False)

# Save best parameters and score to JSON
best_search = {
    "best_value": float(study.best_value),
    "best_params": study.best_params
}
log_json(best_search, OUT_DIR / "cv" / "hpo_best.json")

# Print best result
print("✅ 5-fold CV search completed — Best configuration found:")
print(best_search)


# %%
# ================================================================
# 🏁 FINAL TRAINING: Train the best model once with long patience
# ================================================================

# 1️⃣ Load the best hyperparameters found by Optuna
final_cfg = dict(study.best_params)  # take best params from the hyperparameter search
final_cfg.update({
    "epochs":   EPOCHS_FINAL,   # increase training epochs for final run
    "patience": PATIENCE_FINAL  # increase patience so early stopping is not too aggressive
})

# Save final training config for reference (useful for reproducibility later)
log_json(final_cfg, OUT_DIR / "final" / "final_params.json")

# ------------------------------------------------
# 2️⃣ Create a small validation set (10%) from the training data
# ------------------------------------------------
# We split the original training data (80% of total) into:
# - 90% → actual training (X_sub, y_sub)
# - 10% → validation set for early stopping and plotting (X_val, y_val)

bins_tr = make_bins_for_stratify(y_tr, 10)   # stratify based on y distribution
X_sub, X_val, y_sub, y_val = train_test_split(
    X_tr, y_tr, 
    test_size=0.1,             # 10% for validation
    random_state=SEED,         # reproducibility
    stratify=bins_tr           # stratified split for balanced validation set
)

# ------------------------------------------------
# 3️⃣ Train the final model (now with saving enabled)
# ------------------------------------------------
# cv_mode=False → this time we want:
# - Training/validation loss curves to be plotted
# - Scatter plots and residual histograms saved
# - Best model weights to be stored

model_final, scaler_final, m_val = train_one_fold(
    X_sub, y_sub,             # training data
    X_val, y_val,             # validation data
    cfg={**final_cfg,         # pass in all final hyperparameters
        "use_bn": True,
         "layers": int(final_cfg["layers"]),
         "width": int(final_cfg["width"]),
         "dropout": float(final_cfg["dropout"]),
         "batch_size": int(final_cfg["batch_size"]),
         "loss": final_cfg["loss"],
         "activation": final_cfg["activation"],
         "max_lr": float(final_cfg["max_lr"]),
         "weight_decay": float(final_cfg["weight_decay"]),
         "pct_start": float(final_cfg["pct_start"])},
    device=DEVICE,
    cv_mode=False,            # ⚠️ Important: enables saving plots and model files
    save_prefix=str(OUT_DIR / "final" / "fulltrain")  # prefix for saving plots
)

# ------------------------------------------------
# 4️⃣ Save the final trained model and other artifacts
# ------------------------------------------------
# We now save:
# - The trained model weights (.pt)
# - The scaler (so future inputs can be scaled the same way)
# - The feature column names (so we know input order later)

# Save trained model weights
torch.save(model_final.state_dict(), OUT_DIR / "models" / "final_fulltrain.pt")

# Save the fitted scaler (for preprocessing future data)
joblib.dump(scaler_final, OUT_DIR / "models" / "scaler.joblib")

# Save the list of feature column names used for training
log_json({"feature_cols": feature_cols}, OUT_DIR / "models" / "features.json")

# ------------------------------------------------
# 5️⃣ Final message: Model is ready 🎉
# ------------------------------------------------
print("✅ Final trained model saved in:", (OUT_DIR / "models").resolve())


# %%
# ================================================================
# 🧪 TEST: Evaluate the final model on the test set
# ================================================================

# 1️⃣ Scale the test data using the same scaler used during training
# (Important: never fit a new scaler on test data — only transform!)
Xs_te = scaler_final.transform(X_te).astype(np.float32)

# 2️⃣ Make predictions with the trained model
model_final.eval()  # put model in evaluation mode (turns off dropout, etc.)
with torch.no_grad():  # no gradient calculation needed during inference
    y_pred = model_final(torch.from_numpy(Xs_te).to(DEVICE)).detach().cpu().numpy()

# ------------------------------------------------
# 3️⃣ Calculate test set metrics
# ------------------------------------------------
# metrics_full() calculates:
#  - MAE (Mean Absolute Error)
#  - RMSE (Root Mean Squared Error)
#  - R² (coefficient of determination)
#  - Pearson correlation
#  - Spearman correlation
test_metrics = metrics_full(y_te, y_pred)

# ------------------------------------------------
# 4️⃣ Plot predicted vs true values and residuals
# ------------------------------------------------

# Scatter plot: shows how close predictions are to true values
plot_scatter_corr(
    y_te, y_pred,
    "TEST diff: true vs pred",                         # title
    OUT_DIR / "final" / "test_scatter_diff.png",       # save location
    metrics=test_metrics                               # show metrics on the plot
)

# Residual plot: distribution of errors (true - predicted)
plot_residuals(
    y_te, y_pred, 
    "TEST diff residuals", 
    OUT_DIR / "final" / "test_residuals_diff.png"
)

# ------------------------------------------------
# 5️⃣ Save test results for reproducibility
# ------------------------------------------------
# Save arrays for future analysis
np.save(OUT_DIR / "final" / "ytest.npy", y_te)   # true labels
np.save(OUT_DIR / "final" / "ypred.npy", y_pred) # predictions

# Save metrics in a JSON file
Path(OUT_DIR / "final" / "test_metrics_diff.json").write_text(json.dumps(test_metrics, indent=2))

# Print results to console
print("✅ Test metrics (diff):", test_metrics)
print("📊 Scatter plot saved at:", OUT_DIR / "final" / "test_scatter_diff.png")

# ================================================================
# 📈 OPTIONAL: Compare experimental DDG vs. model predictions vs. FoldX
# ================================================================
if (ex_te is not None) and (fx_te is not None):
    # Ground-truth experimental ΔΔG
    ddg_true = ex_te

    # FoldX predicted ΔΔG
    ddg_fx = fx_te

    # Our model predicts ΔΔG difference, so we add FoldX base prediction
    ddg_pred = y_pred + fx_te  # final reconstructed ΔΔG

    # Compute metrics for both model and FoldX predictions
    m_model = metrics_full(ddg_true, ddg_pred)
    m_fx    = metrics_full(ddg_true, ddg_fx)

    # Save metrics
    Path(OUT_DIR / "final" / "test_metrics_DDG.json").write_text(json.dumps(m_model, indent=2))
    Path(OUT_DIR / "final" / "test_metrics_DDG_FoldX.json").write_text(json.dumps(m_fx, indent=2))

    # ------------------------------------------------
    # 6️⃣ Plot comparison: Experimental vs Predicted (Model & FoldX)
    # ------------------------------------------------
    plt.figure(figsize=(5.5, 5.5))
    ax = plt.gca()

    # Plot model predictions
    ax.scatter(ddg_true, ddg_pred, alpha=0.6,
        label=(f"Modelo\nMAE={m_model['MAE']:.3f} RMSE={m_model['RMSE']:.3f} R²={m_model['R2']:.3f}\n"
               f"r={m_model['Pearson']:.3f}  ρ={m_model['Spearman']:.3f}"),
        marker="o")

    # Plot FoldX predictions
    ax.scatter(ddg_true, ddg_fx, alpha=0.5,
        label=(f"FoldX\nMAE={m_fx['MAE']:.3f} RMSE={m_fx['RMSE']:.3f} R²={m_fx['R2']:.3f}\n"
               f"r={m_fx['Pearson']:.3f}  ρ={m_fx['Spearman']:.3f}"),
        marker="x")

    # Identity line (perfect prediction line)
    lo = min(np.min(ddg_true), np.min(ddg_pred), np.min(ddg_fx))
    hi = max(np.max(ddg_true), np.max(ddg_pred), np.max(ddg_fx))
    pad = 0.2 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--")

    # Plot styling
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("DDG experimental")
    ax.set_ylabel("DDG predicted")
    ax.set_title("TEST: Experimental DDG vs Predicted (Model vs FoldX)")

    # Legend styling
    leg = ax.legend(loc="lower right")
    if leg is not None:
        leg.set_frame_on(False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(OUT_DIR / "final" / "test_scatter_DDG_combined.png", dpi=200)
    plt.close()

    # ------------------------------------------------
    # 7️⃣ Final console summary
    # ------------------------------------------------
    print("✅ DDG metrics (Model):", m_model)
    print("✅ DDG metrics (FoldX):", m_fx)
    print("📊 Combined DDG scatter saved at:", OUT_DIR / "final" / "test_scatter_DDG_combined.png")


# %%
# ================================================================
# 🌍 OPTIONAL: External Prediction on SARS Dataset
# ================================================================
# If you have an external CSV with the same feature columns, you can run the trained model on it.
# This is useful for checking how well the model generalizes to totally unseen data.

SARS_PATH = "S285_subset_fx_corrected_nodeadcol_invexp.csv"
SARS_SEP  = "\t"

if SARS_PATH:
    # 1️⃣ Create output directory for SARS results
    ext_dir = OUT_DIR / "external" / "sars"
    ext_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # 2️⃣ Load trained artifacts: scaler, config, and model weights
    # ------------------------------------------------
    scaler_ext = joblib.load(OUT_DIR / "models" / "scaler.joblib")       # for feature scaling
    final_cfg  = json.loads((OUT_DIR / "final" / "final_params.json").read_text())  # best hyperparameters
    state      = torch.load(OUT_DIR / "models" / "final_fulltrain.pt", map_location=DEVICE)  # trained weights

    # ------------------------------------------------
    # 3️⃣ Load external dataset and check required columns
    # ------------------------------------------------
    ext_df = pd.read_csv(SARS_PATH, sep=SARS_SEP)
    missing = [c for c in feature_cols if c not in ext_df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in SARS data: {missing}")

    has_foldx = "ddG_foldx" in ext_df.columns
    has_exp   = "exp_ddG"   in ext_df.columns

    # Prepare input features and scale them
    X_ext   = ext_df[feature_cols].to_numpy(np.float32)
    X_ext_s = scaler_ext.transform(X_ext).astype(np.float32)

    # ------------------------------------------------
    # 4️⃣ Rebuild the model and load trained weights
    # ------------------------------------------------
    model_ext = MLP(
        in_dim=len(feature_cols),
        layers=int(final_cfg["layers"]),
        width=int(final_cfg["width"]),
        dropout=float(final_cfg["dropout"]),
        activation=str(final_cfg.get("activation", "relu")),
        use_bn=True
    ).to(DEVICE)

    model_ext.load_state_dict(state)
    model_ext.eval()

    # ------------------------------------------------
    # 5️⃣ Predict the ΔΔG difference (diff) on external data
    # ------------------------------------------------
    with torch.no_grad():
        diff_pred_model = model_ext(torch.from_numpy(X_ext_s).to(DEVICE)).detach().cpu().numpy().ravel()

    # ------------------------------------------------
    # 6️⃣ Prepare prediction output DataFrame
    # ------------------------------------------------
    out = {
        "row_id": np.arange(len(diff_pred_model), dtype=int),
        "diff_pred_model": diff_pred_model.astype(float)
    }
    if has_foldx:
        out["ddG_foldx"] = ext_df["ddG_foldx"].to_numpy(np.float32).astype(float)
    if has_exp:
        out["ddG_exp"] = ext_df["exp_ddG"].to_numpy(np.float32).astype(float)
    if has_foldx and has_exp:
        out["diff_true"] = (
            ext_df["exp_ddG"].to_numpy(np.float32) - ext_df["ddG_foldx"].to_numpy(np.float32)
        ).astype(float)
    if has_foldx:
        # Predicted absolute ΔΔG (FoldX + predicted difference)
        out["ddG_pred_model"] = (diff_pred_model + ext_df["ddG_foldx"].to_numpy(np.float32)).astype(float)

    # Residuals (error = true - predicted)
    if "diff_true" in out:
        out["resid_diff_model"] = (out["diff_true"] - out["diff_pred_model"]).astype(float)
    if ("ddG_exp" in out) and ("ddG_pred_model" in out):
        out["resid_DDG_model"] = (out["ddG_exp"] - out["ddG_pred_model"]).astype(float)
    if ("ddG_exp" in out) and ("ddG_foldx" in out):
        out["resid_DDG_foldx"] = (out["ddG_exp"] - out["ddG_foldx"]).astype(float)

    # Save prediction CSV
    pd.DataFrame(out).to_csv(ext_dir / "sars_predictions.csv", index=False)
    print("✅ SARS prediction CSV saved:", ext_dir / "sars_predictions.csv")

    # ------------------------------------------------
    # 7️⃣ Evaluate metrics on "diff" (if true values available)
    # ------------------------------------------------
    if "diff_true" in out:
        y_true_diff = np.asarray(out["diff_true"], float)
        y_pred_diff = np.asarray(out["diff_pred_model"], float)
        m_diff = metrics_full(y_true_diff, y_pred_diff)

        # Save metrics and plots
        Path(ext_dir / "sars_metrics_diff.json").write_text(json.dumps(m_diff, indent=2))
        plot_scatter_corr(y_true_diff, y_pred_diff, "SARS diff: true vs pred", ext_dir / "sars_scatter_diff.png", metrics=m_diff)
        plot_residuals(y_true_diff, y_pred_diff, "SARS diff residuals", ext_dir / "sars_residuals_diff.png")

        print("📊 SARS diff metrics:", m_diff)

    # ------------------------------------------------
    # 8️⃣ Compare full ΔΔG predictions with FoldX (if available)
    # ------------------------------------------------
    if has_exp and has_foldx and "ddG_pred_model" in out:
        ddg_true = np.asarray(out["ddG_exp"], float)
        ddg_pred = np.asarray(out["ddG_pred_model"], float)
        ddg_fx   = np.asarray(out["ddG_foldx"], float)

        m_model = metrics_full(ddg_true, ddg_pred)
        m_fx    = metrics_full(ddg_true, ddg_fx)

        # Save metrics
        Path(ext_dir / "sars_metrics_DDG_model.json").write_text(json.dumps(m_model, indent=2))
        Path(ext_dir / "sars_metrics_DDG_FoldX.json").write_text(json.dumps(m_fx, indent=2))

        # Combined scatter plot: Experimental vs Model vs FoldX
        plt.figure(figsize=(5.5, 5.5))
        ax = plt.gca()

        lo = float(np.min([ddg_true, ddg_pred, ddg_fx]))
        hi = float(np.max([ddg_true, ddg_pred, ddg_fx]))
        pad = 0.2 * (hi - lo if hi > lo else 1.0)

        # Plot model predictions
        ax.scatter(
            ddg_true, ddg_pred, alpha=0.6,
            label=f"Model\nMAE={m_model['MAE']:.3f} RMSE={m_model['RMSE']:.3f} R²={m_model['R2']:.3f}\n"
                  f"r={m_model['Pearson']:.3f}  ρ={m_model['Spearman']:.3f}"
        )
        # Plot FoldX predictions
        ax.scatter(
            ddg_true, ddg_fx, alpha=0.5,
            label=f"FoldX\nMAE={m_fx['MAE']:.3f} RMSE={m_fx['RMSE']:.3f} R²={m_fx['R2']:.3f}\n"
                  f"r={m_fx['Pearson']:.3f}  ρ={m_fx['Spearman']:.3f}"
        )

        # Identity line (perfect prediction)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Experimental ΔΔG")
        ax.set_ylabel("Predicted ΔΔG")
        ax.set_title("SARS: Experimental vs Predicted ΔΔG (Model vs FoldX)")

        # Style legend
        leg = ax.legend(loc="lower right")
        if leg is not None:
            leg.set_frame_on(False)

        plt.tight_layout()
        plt.savefig(ext_dir / "sars_scatter_DDG_combined.png", dpi=200)
        plt.close()

        print("📊 SARS ΔΔG metrics (Model):", m_model)
        print("📊 SARS ΔΔG metrics (FoldX):", m_fx)
        print("📈 Combined scatter plot saved:", ext_dir / "sars_scatter_DDG_combined.png")

    # ------------------------------------------------
    # 9️⃣ Final confirmation
    # ------------------------------------------------
    print("✅ All SARS artifacts saved in:", ext_dir)


# %%



