"""Dataset loading and preprocessing: raw OULAD (zip or extracted folder)
and UCI Student Performance, plus group-aware/plain train-test splitting,
a fairness diagnostics report, and a synthetic demo dataset generator.
"""
import numpy as np
import pandas as pd


def _read_oulad_member(zip_path, member, **kwargs):
    """Read one CSV from an OULAD folder"""
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        match = next((n for n in names if n.replace("\\", "/").split("/")[-1].lower() == member.lower()), None)
        if match is None:
            raise FileNotFoundError(f"OULAD archive does not contain {member}.")
        with z.open(match) as f:
            return pd.read_csv(f, **kwargs)


def _find_oulad_file(directory, filename):
    import os
    direct = os.path.join(directory, filename)
    if os.path.exists(direct):
        return direct
    # Also allow the common extracted folder name "OULAD data".
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not find {filename} below {directory}")


def _build_oulad_features(source_path, early_days=28, chunksize=250000):
    """Build one row per student/module/presentation from raw OULAD.
    """
    import os
    import zipfile

    if os.path.isfile(source_path) and source_path.lower().endswith('.zip'):
        student_info = _read_oulad_member(source_path, 'studentInfo.csv')
        assessments = _read_oulad_member(source_path, 'assessments.csv')
        registration = _read_oulad_member(source_path, 'studentRegistration.csv')
        vle = _read_oulad_member(source_path, 'vle.csv')

        # Assessment metadata available early in the presentation.
        early_assess = assessments[
            assessments["date"].notna() & (assessments["date"] <= early_days)
        ].copy()
        early_assess = early_assess[["id_assessment", "code_module", "code_presentation", "weight", "date"]]

        # Student assessment results, joined only to early assessments.
        sa = _read_oulad_member(source_path, 'studentAssessment.csv')
        sa = sa.merge(early_assess, on="id_assessment", how="inner")
        sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
        sa["weight"] = pd.to_numeric(sa["weight"], errors="coerce").fillna(0.0)
        sa["weighted_score"] = sa["score"] * sa["weight"]
        ass = sa.groupby(["id_student", "code_module", "code_presentation"], as_index=False).agg(
            mean_assessment_score=("score", "mean"),
            std_assessment_score=("score", "std"),
            assessments_completed=("id_assessment", "count"),
            weighted_score_sum=("weighted_score", "sum"),
            assessment_weight_sum=("weight", "sum"),
        )
        ass["std_assessment_score"] = ass["std_assessment_score"].fillna(0.0)
        ass["weighted_assessment_score"] = np.where(
            ass["assessment_weight_sum"] > 0,
            ass["weighted_score_sum"] / ass["assessment_weight_sum"],
            ass["mean_assessment_score"]
        )
        # Submission lateness relative to assessment date is observable by day 28.
        sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
        sa["late_flag"] = ((sa["date_submitted"] > sa["date"]) & (sa["date_submitted"] <= early_days)).astype(int)
        late = sa.groupby(["id_student", "code_module", "code_presentation"], as_index=False)["late_flag"].sum()
        ass = ass.merge(late, on=["id_student", "code_module", "code_presentation"], how="left")
        ass = ass.rename(columns={"late_flag": "late_submissions"})
        ass = ass.drop(columns=["weighted_score_sum", "assessment_weight_sum"])

        # VLE aggregation in chunks because studentVle.csv is very large.
        vle_site = vle[["id_site", "activity_type"]].drop_duplicates("id_site")
        chunks = []
        with zipfile.ZipFile(source_path) as z:
            member = next(n for n in z.namelist() if n.lower().endswith('studentvle.csv'))
            with z.open(member) as f:
                for chunk in pd.read_csv(f, chunksize=chunksize):
                    chunk = chunk[chunk["date"].notna() & (chunk["date"] <= early_days)].copy()
                    if chunk.empty:
                        continue
                    chunk = chunk.merge(vle_site, on="id_site", how="left")
                    chunk["sum_click"] = pd.to_numeric(chunk["sum_click"], errors="coerce").fillna(0.0)
                    g = chunk.groupby(["id_student", "code_module", "code_presentation"], as_index=False).agg(
                        total_clicks=("sum_click", "sum"),
                        vle_events=("id_site", "count"),
                        active_days=("date", "nunique"),
                        active_weeks=("date", lambda x: int(np.ceil((pd.to_numeric(x).max() + 1) / 7.0)) if len(x) else 0),
                    )
                    # Activity-type clicks provide a small interpretable subset.
                    pivot = chunk.pivot_table(
                        index=["id_student", "code_module", "code_presentation"],
                        columns="activity_type", values="sum_click", aggfunc="sum", fill_value=0
                    ).reset_index()
                    pivot.columns = [
                        c if isinstance(c, str) else str(c) for c in pivot.columns
                    ]
                    pivot = pivot.rename(columns={c: f"clicks_{c}" for c in pivot.columns
                                                   if c not in ["id_student", "code_module", "code_presentation"]})
                    g = g.merge(pivot, on=["id_student", "code_module", "code_presentation"], how="left")
                    chunks.append(g)

        if chunks:
            vle_agg = pd.concat(chunks, ignore_index=True)
            numeric = [c for c in vle_agg.columns if c.startswith("clicks_") or c in ["total_clicks", "vle_events", "active_days", "active_weeks"]]
            vle_agg[numeric] = vle_agg[numeric].fillna(0.0)
            vle_agg = vle_agg.groupby(["id_student", "code_module", "code_presentation"], as_index=False)[numeric].sum()
        else:
            vle_agg = pd.DataFrame(columns=["id_student", "code_module", "code_presentation", "total_clicks", "vle_events", "active_days", "active_weeks"])

    else:
        # Extracted directory version.
        student_info = pd.read_csv(_find_oulad_file(source_path, "studentInfo.csv"))
        assessments = pd.read_csv(_find_oulad_file(source_path, "assessments.csv"))
        registration = pd.read_csv(_find_oulad_file(source_path, "studentRegistration.csv"))
        vle = pd.read_csv(_find_oulad_file(source_path, "vle.csv"))
        early_assess = assessments[assessments["date"].notna() & (assessments["date"] <= early_days)].copy()
        sa = pd.read_csv(_find_oulad_file(source_path, "studentAssessment.csv"))
        sa = sa.merge(early_assess[["id_assessment", "code_module", "code_presentation", "weight", "date"]], on="id_assessment", how="inner")
        sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
        sa["weight"] = pd.to_numeric(sa["weight"], errors="coerce").fillna(0.0)
        sa["weighted_score"] = sa["score"] * sa["weight"]
        ass = sa.groupby(["id_student", "code_module", "code_presentation"], as_index=False).agg(
            mean_assessment_score=("score", "mean"), std_assessment_score=("score", "std"),
            assessments_completed=("id_assessment", "count"), weighted_score_sum=("weighted_score", "sum"),
            assessment_weight_sum=("weight", "sum"))
        ass["std_assessment_score"] = ass["std_assessment_score"].fillna(0.0)
        ass["weighted_assessment_score"] = np.where(ass["assessment_weight_sum"] > 0, ass["weighted_score_sum"] / ass["assessment_weight_sum"], ass["mean_assessment_score"])
        sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
        sa["late_flag"] = ((sa["date_submitted"] > sa["date"]) & (sa["date_submitted"] <= early_days)).astype(int)
        late = sa.groupby(["id_student", "code_module", "code_presentation"], as_index=False)["late_flag"].sum()
        ass = ass.merge(late, on=["id_student", "code_module", "code_presentation"], how="left").rename(columns={"late_flag": "late_submissions"})
        ass = ass.drop(columns=["weighted_score_sum", "assessment_weight_sum"])
        sv = pd.read_csv(_find_oulad_file(source_path, "studentVle.csv"), usecols=["code_module","code_presentation","id_student","id_site","date","sum_click"])
        sv = sv[sv["date"].notna() & (sv["date"] <= early_days)].copy().merge(vle[["id_site","activity_type"]], on="id_site", how="left")
        sv["sum_click"] = pd.to_numeric(sv["sum_click"], errors="coerce").fillna(0.0)
        vle_agg = sv.groupby(["id_student","code_module","code_presentation"], as_index=False).agg(total_clicks=("sum_click","sum"), vle_events=("id_site","count"), active_days=("date","nunique"))
        vle_agg["active_weeks"] = np.ceil((sv.groupby(["id_student","code_module","code_presentation"])["date"].max().reset_index(name="max_date")["max_date"] + 1) / 7.0).values
        pivot = sv.pivot_table(index=["id_student","code_module","code_presentation"], columns="activity_type", values="sum_click", aggfunc="sum", fill_value=0).reset_index()
        pivot.columns = [c if isinstance(c,str) else str(c) for c in pivot.columns]
        pivot = pivot.rename(columns={c:f"clicks_{c}" for c in pivot.columns if c not in ["id_student","code_module","code_presentation"]})
        vle_agg = vle_agg.merge(pivot, on=["id_student","code_module","code_presentation"], how="left")

    # Registration: use only date_registration; date_unregistration is post-outcome leakage.
    reg = registration[["id_student", "code_module", "code_presentation", "date_registration"]].copy()
    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    reg["registration_days_before_start"] = -reg["date_registration"]
    reg = reg.drop(columns=["date_registration"])

    base = student_info.copy()
    base["at_risk"] = base["final_result"].isin(["Fail", "Withdrawn"]).astype(np.float64)
    base = base.drop(columns=["final_result"])
    base = base.merge(ass, on=["id_student","code_module","code_presentation"], how="left")
    base = base.merge(vle_agg, on=["id_student","code_module","code_presentation"], how="left")
    base = base.merge(reg, on=["id_student","code_module","code_presentation"], how="left")

    numeric_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if c not in ["id_student", "at_risk"]:
            base[c] = base[c].fillna(0.0)

    # Per-active-period normalisations.
    base["mean_daily_clicks"] = base["total_clicks"] / np.maximum(base["active_days"], 1)
    base["clicks_per_active_week"] = base["total_clicks"] / np.maximum(base["active_weeks"], 1)

    # Keep an ID for group-aware train/test splitting, but NEVER expose it to agents.
    groups = base["id_student"].astype(str).values
    protected = {}
    for attr in ["gender", "age_band", "disability", "region", "highest_education", "imd_band"]:
        if attr in base.columns:
            protected[attr] = base[attr].fillna("Unknown").astype(str).values

    protected["gender"] = protected.get("gender", np.array(["Unknown"] * len(base)))
    protected["disability"] = protected.get("disability", np.array(["Unknown"] * len(base)))

    # Explicit feature groups for heterogeneous agents.
    academic = ["mean_assessment_score", "std_assessment_score", "assessments_completed",
                "weighted_assessment_score", "late_submissions"]
    engagement = ["total_clicks", "vle_events", "active_days", "active_weeks",
                  "mean_daily_clicks", "clicks_per_active_week"]
    behaviour = engagement + [c for c in base.columns if c.startswith("clicks_")]
    demographic = ["num_of_prev_attempts", "studied_credits", "registration_days_before_start"]

    # Drop identifiers and target before one-hot encoding.
    X_raw = base.drop(columns=["at_risk", "id_student"])
    X_enc = pd.get_dummies(X_raw, columns=[c for c in X_raw.columns if not pd.api.types.is_numeric_dtype(X_raw[c])], drop_first=False)
    feature_names = X_enc.columns.tolist()
    X = X_enc.astype(np.float64).values
    feature_index = {f: i for i, f in enumerate(feature_names)}

    def resolve(cols):
        return [feature_index[c] for c in cols if c in feature_index]

    academic_idx = resolve(academic)
    engagement_idx = resolve(engagement)
    behaviour_idx = resolve(behaviour)
    demographic_idx = resolve(demographic)
    full_idx = list(range(X.shape[1]))

    # For report/UI labelling, include categories of the specialised partitions.
    feature_groups = {
        "academic_idx": academic_idx,
        "behaviour_idx": behaviour_idx,
        "engagement_idx": engagement_idx,
        "demographic_idx": demographic_idx,
        "full_idx": full_idx,
    }
    return X, base["at_risk"].values.astype(np.float64), feature_names, feature_groups, protected, groups


def _validate_oulad_schema(source_path):
    """Checks that each required OULAD CSV has the columns
    _build_oulad_features actually needs, BEFORE any of the merging/
    aggregation logic runs."""
    required_columns = {
        "studentInfo.csv": ["id_student", "code_module", "code_presentation", "final_result"],
        "assessments.csv": ["id_assessment", "code_module", "code_presentation", "weight", "date"],
        "studentAssessment.csv": ["id_assessment", "id_student", "score", "date_submitted"],
        "studentVle.csv": ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        "vle.csv": ["id_site", "activity_type"],
        "studentRegistration.csv": ["id_student", "code_module", "code_presentation", "date_registration"],
    }
    problems = []
    for filename, cols in required_columns.items():
        try:
            path = _find_oulad_file(source_path, filename)
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception as exc:
            problems.append(f"{filename}: could not read header ({exc})")
            continue
        missing = [c for c in cols if c not in header]
        if missing:
            problems.append(f"{filename}: missing column(s) {missing}")
    if problems:
        raise ValueError("OULAD folder has schema problems:\n  " + "\n  ".join(problems))


def load_and_preprocess(filepath: str, target_col: str = "G3", at_risk_threshold: float = 10,
                        expected_n_columns: int = 33, dataset_type: str = "auto"):
    """Load either the raw UCI Student Performance CSV or raw OULAD data.
    """
    import os
    lower = str(filepath).lower()

    if dataset_type not in {"auto", "uci", "oulad"}:
        raise ValueError("dataset_type must be 'uci', 'oulad', or 'auto'.")

    # Explicit OULAD mode accepts ONLY the extracted raw OULAD directory.
    # ZIP and pre-generated flat OULAD feature files are intentionally not
    # accepted by the new dataset-selection workflow.
    if dataset_type == "oulad":
        if not os.path.isdir(filepath):
            raise ValueError(
                "OULAD expects the extracted raw-data folder containing "
                "studentInfo.csv, assessments.csv, studentAssessment.csv, "
                "studentVle.csv, studentRegistration.csv, and vle.csv."
            )
        required = ["studentInfo.csv", "assessments.csv", "studentAssessment.csv",
                    "studentVle.csv", "studentRegistration.csv", "vle.csv"]
        missing = []
        for name in required:
            try:
                _find_oulad_file(filepath, name)
            except FileNotFoundError:
                missing.append(name)
        if missing:
            raise FileNotFoundError("OULAD folder is missing: " + ", ".join(missing))
        _validate_oulad_schema(filepath)
        return _build_oulad_features(filepath)

    # Auto mode is retained for backwards compatibility with scripts that
    # call load_and_preprocess() directly. It recognizes raw OULAD folders.
    if dataset_type == "auto" and os.path.isdir(filepath) and os.path.exists(os.path.join(filepath, "studentInfo.csv")):
        return _build_oulad_features(filepath)

    if dataset_type == "oulad":
        raise ValueError("Invalid raw OULAD directory.")

    # UCI mode: raw student-mat.csv/student-por.csv style file.
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"UCI dataset file not found: {filepath}")
    df = pd.read_csv(filepath, sep=";")
    if expected_n_columns is not None and df.shape[1] != expected_n_columns:
        raise ValueError(f"Dataset must contain exactly {expected_n_columns} columns (found {df.shape[1]}).")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")
    y = (df[target_col] < at_risk_threshold).astype(int).values.astype(np.float64)
    X_raw = df.drop(columns=[target_col])
    X_enc = pd.get_dummies(X_raw, drop_first=False)
    feature_names = X_enc.columns.tolist()
    X = X_enc.values.astype(np.float64)
    grade_idx = [feature_names.index(f) for f in ["G1", "G2"] if f in feature_names]
    behaviour_idx = [feature_names.index(f) for f in ["absences", "failures", "studytime", "goout", "Dalc", "Walc", "health"] if f in feature_names]
    full_idx = list(range(X.shape[1]))
    protected = {a: df[a].values for a in ("sex", "school") if a in df.columns}
    groups = df["id_student"].astype(str).values if "id_student" in df.columns else None
    groups_dict = {"academic_idx": grade_idx, "behaviour_idx": behaviour_idx,
                   "engagement_idx": behaviour_idx, "demographic_idx": [], "full_idx": full_idx}
    return X, y, feature_names, groups_dict, protected, groups

def train_test_split(X, y, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def group_train_test_split(X, y, groups, test_frac=0.25, seed=0):
    """Group-aware split used for OULAD so the same student never appears in
    both training and test sets. This prevents identity-level leakage across
    different module/presentation records."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    unique_groups = unique_groups.copy()
    rng.shuffle(unique_groups)
    n_test_groups = max(1, int(len(unique_groups) * test_frac))
    test_groups = set(unique_groups[:n_test_groups])
    test_mask = np.array([g in test_groups for g in groups])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def fairness_report(y_true, y_pred, protected_attributes):
    """Group fairness diagnostics. Positive class is At Risk (1).
    Reports accuracy, precision, recall/TPR, FPR and positive prediction rate
    per protected group plus max/min gaps."""
    rows = []
    if not protected_attributes:
        return pd.DataFrame(rows)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    for attr, values in protected_attributes.items():
        values = np.asarray(values)
        for group in np.unique(values):
            m = values == group
            yt, yp = y_true[m], y_pred[m]
            tp = np.sum((yt == 1) & (yp == 1)); tn = np.sum((yt == 0) & (yp == 0))
            fp = np.sum((yt == 0) & (yp == 1)); fn = np.sum((yt == 1) & (yp == 0))
            rows.append({
                "attribute": attr, "group": str(group), "n": int(m.sum()),
                "accuracy": float((tp + tn) / len(yt)) if len(yt) else 0.0,
                "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                "recall_tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
                "positive_rate": float(yp.mean()) if len(yp) else 0.0,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        # One compact GAP(max-min) row per protected attribute, summarizing
        # the spread across its groups on each metric.
        gap_rows = []
        for attr, gdf in df.groupby("attribute"):
            gap_rows.append({"attribute": attr, "group": "GAP(max-min)", "n": np.nan,
                             "accuracy": gdf["accuracy"].max() - gdf["accuracy"].min(),
                             "precision": gdf["precision"].max() - gdf["precision"].min(),
                             "recall_tpr": gdf["recall_tpr"].max() - gdf["recall_tpr"].min(),
                             "fpr": gdf["fpr"].max() - gdf["fpr"].min(),
                             "positive_rate": gdf["positive_rate"].max() - gdf["positive_rate"].min()})
        df = pd.concat([df, pd.DataFrame(gap_rows)], ignore_index=True)
    return df


def make_synthetic_dataset(n_samples=300, n_features=8, seed=42):
    rng = np.random.default_rng(seed)
    n_class1 = n_samples // 2
    n_class0 = n_samples - n_class1
    X0 = rng.normal(loc=0.0, scale=1.0, size=(n_class0, n_features))
    X1 = rng.normal(loc=1.8, scale=1.1, size=(n_class1, n_features))
    X0[:, 2:] *= 0.5
    X1[:, 2:] *= 0.5
    X = np.vstack([X0, X1])
    y = np.array([0] * n_class0 + [1] * n_class1)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


