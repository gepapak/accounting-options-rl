"""FIT-only supervised audit critics; predictions are in original money (DKK).

No model is fitted on import. These predictors are nuisance controls, not
validated reward/transition models. Certificate validity must come from the
caller's common-branch centering, amplitude budget and independent confirmation.
Only trusted artifacts created by this module may be loaded with pickle.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time
import warnings

SCHEMA = "sample_only_audit_critics_v1"
BUNDLE_NAME = "audit_models.pkl"
METADATA_NAME = "audit_models_metadata.json"
MANIFEST_NAME = "audit_models_manifest.json"
PREDICTION_BATCH_SIZE = 4096


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")


def _feature_matrix(X, n_features=None, allow_empty=False):
    import numpy as np
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("X must be a two-dimensional numeric feature matrix")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("X has no rows")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, received {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ValueError("X contains nonfinite features")
    return array


def _array_digest(array):
    import numpy as np
    value = np.ascontiguousarray(array, dtype="<f8")
    header = _json_bytes(dict(shape=list(value.shape), dtype="little-endian float64"))
    return _sha_bytes(header+value.tobytes(order="C"))


def _scaler_metadata(scaler):
    import numpy as np
    return dict(with_mean=scaler.with_mean, with_std=scaler.with_std,
                mean=scaler.mean_.tolist(), variance=scaler.var_.tolist(),
                scale=scaler.scale_.tolist(),
                n_samples_seen=np.asarray(scaler.n_samples_seen_).tolist())


def _finite_number(value):
    import math
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class MoneyPredictor:
    """Read-only prediction interface; it never fits or updates a normalizer."""

    _estimator: object
    _x_scaler: object
    _y_scaler: object
    n_features: int
    name: str

    def predict(self, X):
        import numpy as np
        from threadpoolctl import threadpool_limits

        matrix = _feature_matrix(X, self.n_features, allow_empty=True)
        result = np.empty(matrix.shape[0], dtype=np.float64)
        # Fixed batches and one numerical thread: no stochastic inference,
        # refitting, endpoint queries or checkpoint normalization changes.
        with threadpool_limits(limits=1):
            for start in range(0, matrix.shape[0], PREDICTION_BATCH_SIZE):
                stop = min(start+PREDICTION_BATCH_SIZE, matrix.shape[0])
                standardized_x = self._x_scaler.transform(matrix[start:stop])
                standardized_y = np.asarray(self._estimator.predict(standardized_x), dtype=np.float64)
                result[start:stop] = self._y_scaler.inverse_transform(standardized_y.reshape(-1, 1)).ravel()
        if not np.isfinite(result).all():
            raise ValueError(f"{self.name} produced a nonfinite money prediction")
        return result


def fit_models(X, y, output_dir, seed=39300000,target_label="GLOBAL discounted future NEW paired-injection-minus-cost"):
    """Fit ridge and MLP on caller-supplied FIT rows, save them, return metadata.

    X contains actual-branch features only. y is the caller's GLOBAL discounted
    future paired-injection-minus-cost target in DKK. The caller owns feature
    construction, FIT/SEARCH/CONFIRM separation and target correctness. This
    function neither loads simulator data nor queries policy endpoints.
    """
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    import numpy as np
    import sklearn
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    directory = Path(output_dir)
    artifacts = tuple(directory/name for name in (BUNDLE_NAME, METADATA_NAME, MANIFEST_NAME))
    if any(path.exists() for path in artifacts):
        raise FileExistsError("Preserve existing/partial critic artifacts; use load_models or a new output directory")
    matrix = _feature_matrix(X)
    target = np.asarray(y, dtype=np.float64)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target.ravel()
    if target.ndim != 1 or len(target) != len(matrix) or not np.isfinite(target).all():
        raise ValueError("y must be a finite money-valued vector with one target per FIT row")
    if len(target) < 10:
        raise ValueError("At least 10 FIT rows are required for the fixed 20% early-stopping validation split")
    seed = int(seed)
    preprocess_cpu = time.process_time()
    clip_low, clip_high = np.quantile(target, (0.01, 0.99), method="linear")
    clipped_target = np.clip(target, clip_low, clip_high)
    x_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    y_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    standardized_x = x_scaler.fit_transform(matrix)
    standardized_y = y_scaler.fit_transform(clipped_target.reshape(-1, 1)).ravel()
    if not np.isfinite(standardized_x).all() or not np.isfinite(standardized_y).all():
        raise ValueError("FIT-only normalization produced nonfinite values")
    preprocess_cpu = time.process_time()-preprocess_cpu
    ridge = Ridge(alpha=1.0, random_state=seed)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
        alpha=1e-3, learning_rate_init=1e-3, max_iter=120, early_stopping=True,
        validation_fraction=0.2, n_iter_no_change=12, batch_size=256, random_state=seed)
    fit_times = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with threadpool_limits(limits=1):
            for name, estimator in (("ridge", ridge), ("mlp", mlp)):
                before_cpu, before_wall = time.process_time(), time.perf_counter()
                estimator.fit(standardized_x, standardized_y)
                fit_times[name] = dict(cpu_seconds=time.process_time()-before_cpu,
                                       wall_seconds=time.perf_counter()-before_wall)
            # Descriptive FIT loss only; this is not an endpoint/confirmation
            # query and is never used to tune or select a model.
            before_diagnostics = time.process_time()
            ridge_fit_residual = ridge.predict(standardized_x)-standardized_y
            ridge_fit_mse = float(np.mean(ridge_fit_residual**2))
            ridge_diagnostic_cpu = time.process_time()-before_diagnostics
    warning_records = [dict(category=item.category.__name__, message=str(item.message)) for item in caught]
    bundle = dict(schema=SCHEMA, n_features=int(matrix.shape[1]), x_scaler=x_scaler,
                  y_scaler=y_scaler, ridge=ridge, mlp=mlp)
    bundle_bytes = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = dict(schema=SCHEMA, seed=seed, money_units="DKK", training_split="FIT only",
        target=str(target_label),
        feature_construction="caller-supplied actual-branch features; no endpoint queries",
        n_rows=int(matrix.shape[0]), n_features=int(matrix.shape[1]), numerical_threads=1,
        training_input_sha256=_array_digest(matrix), raw_target_sha256=_array_digest(target),
        clipped_target_sha256=_array_digest(clipped_target),
        target_robustification=dict(quantiles=[0.01, 0.99], method="linear", fit_rows_only=True,
            clip_lower=float(clip_low), clip_upper=float(clip_high),
            n_clipped_below=int(np.sum(target < clip_low)), n_clipped_above=int(np.sum(target > clip_high)),
            raw_min=float(target.min()), raw_max=float(target.max()),
            order="clip FIT target, then fit target StandardScaler on clipped FIT target"),
        x_scaler=_scaler_metadata(x_scaler), y_scaler=_scaler_metadata(y_scaler),
        ridge=dict(parameters=ridge.get_params(deep=False), **fit_times["ridge"],
            n_iter=None if ridge.n_iter_ is None else np.asarray(ridge.n_iter_).tolist(),
            fit_mse_standardized_clipped_target=ridge_fit_mse,
            fit_mse_clipped_target_DKK_squared=ridge_fit_mse*float(y_scaler.scale_[0])**2,
            fit_loss_diagnostic_cpu_seconds=ridge_diagnostic_cpu),
        mlp=dict(parameters=mlp.get_params(deep=False), **fit_times["mlp"],
            n_iter=int(mlp.n_iter_), training_samples_seen=int(mlp.t_),
            loss_scaled_target=_finite_number(mlp.loss_),
            loss_curve_scaled_target=[_finite_number(v) for v in mlp.loss_curve_],
            validation_scores=[_finite_number(v) for v in (mlp.validation_scores_ or [])],
            best_validation_score=_finite_number(mlp.best_validation_score_),
            validation_scope="internal randomized 20% subset of FIT; scalers see FIT only, including this subset"),
        preprocessing_cpu_seconds=preprocess_cpu, total_cpu_seconds=time.process_time()-cpu_start,
        total_wall_seconds=time.perf_counter()-wall_start, prediction_batch_size=PREDICTION_BATCH_SIZE,
        versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
        warnings=warning_records, source_sha256=_sha_bytes(Path(__file__).read_bytes()),
        bundle_file=BUNDLE_NAME, bundle_sha256=_sha_bytes(bundle_bytes),
        validity="Predictive accuracy is not a certificate assumption; caller must impose exact sign centering and its pathwise budget")
    metadata_bytes = _json_bytes(metadata)
    manifest = dict(schema=SCHEMA, complete=True, bundle_file=BUNDLE_NAME,
        bundle_sha256=_sha_bytes(bundle_bytes), metadata_file=METADATA_NAME,
        metadata_sha256=_sha_bytes(metadata_bytes), source_sha256=metadata["source_sha256"])
    manifest_bytes = _json_bytes(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves partial/completed artifacts. The manifest is
    # the final completion marker; loading requires and verifies both hashes.
    for path, content in zip(artifacts, (bundle_bytes, metadata_bytes, manifest_bytes)):
        with path.open("xb") as stream:
            stream.write(content)
    return dict(metadata, metadata_sha256=manifest["metadata_sha256"],
                manifest_sha256=_sha_bytes(manifest_bytes), artifact_directory=str(directory.resolve()))


def load_models(output_dir):
    """Load our TRUSTED saved critics after checking metadata/bundle hashes."""
    directory = Path(output_dir)
    manifest = json.loads((directory/MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("Missing/incompatible completed audit-critic manifest")
    bundle_bytes = (directory/BUNDLE_NAME).read_bytes()
    metadata_bytes = (directory/METADATA_NAME).read_bytes()
    if _sha_bytes(bundle_bytes) != manifest["bundle_sha256"]:
        raise ValueError("Saved audit-critic pickle hash mismatch")
    if _sha_bytes(metadata_bytes) != manifest["metadata_sha256"]:
        raise ValueError("Saved audit-critic metadata hash mismatch")
    metadata = json.loads(metadata_bytes)
    if metadata.get("schema") != SCHEMA or metadata["bundle_sha256"] != manifest["bundle_sha256"]:
        raise ValueError("Saved audit-critic metadata is inconsistent")
    # Never pass downloaded, user-supplied or otherwise untrusted pickle files.
    bundle = pickle.loads(bundle_bytes)
    if bundle.get("schema") != SCHEMA or bundle["n_features"] != metadata["n_features"]:
        raise ValueError("Saved audit-critic bundle is inconsistent")
    return {name: MoneyPredictor(bundle[name], bundle["x_scaler"], bundle["y_scaler"],
                int(bundle["n_features"]), name) for name in ("ridge", "mlp")}
