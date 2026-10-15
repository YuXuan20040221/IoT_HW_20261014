import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy.stats import jarque_bera
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

ID_COLUMN_HINTS = {"id", "idx", "index", "unnamed: 0", "unnamed_0"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working.columns = [str(col).strip() for col in working.columns]
    seen = {}
    normalized = []
    for col in working.columns:
        count = seen.get(col, 0)
        seen[col] = count + 1
        normalized.append(col if count == 0 else f"{col}_{count}")
    working.columns = normalized
    return working


def prepare_feature_matrix(df: pd.DataFrame, target_col: str):
    feature_table = pd.DataFrame(index=df.index)
    skipped = []

    for col in df.columns:
        if col == target_col:
            continue
        if col.lower() in ID_COLUMN_HINTS:
            skipped.append(col)
            continue

        series = df[col].copy()
        if pd.api.types.is_numeric_dtype(series):
            feature_table[col] = pd.to_numeric(series, errors="coerce")
            continue

        dt_values = pd.to_datetime(series, errors="coerce")
        if dt_values.notna().sum() >= max(1, int(len(series) * 0.8)):
            feature_table[f"{col}_ordinal"] = dt_values.map(pd.Timestamp.toordinal)
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue
        if non_null.astype(str).nunique() <= 20:
            dummies = pd.get_dummies(series.astype(str), prefix=col)
            feature_table = pd.concat([feature_table, dummies], axis=1)
        else:
            skipped.append(col)

    return feature_table, skipped


st.set_page_config(page_title="CSV 回歸分析工具", page_icon="📊", layout="wide")
st.title("📊 CSV 回歸分析工具")
st.caption("自動讀取 CSV · 選擇目標與特徵 · 線性回歸診斷 · 匯出預測結果")

df = None
source_label = "尚未提供資料"
numeric_targets = []
target_column = None
feature_frame = pd.DataFrame()
available_features = []
selected_features = []
skipped_columns = []
use_kbest = False
k_features = None
test_size = 0.2
random_state = 42

with st.sidebar:
    st.header("資料與模型設定")
    uploaded = st.file_uploader("上傳 CSV", type=["csv"])
    default_path = Path("data/train.csv")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_label = "上傳的 CSV"
    elif default_path.exists():
        df = pd.read_csv(default_path)
        source_label = str(default_path)
    else:
        df = None
        source_label = "尚未提供資料"

    if df is not None:
        df = normalize_columns(df)
        numeric_targets = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in ID_COLUMN_HINTS
        ]
        if numeric_targets:
            target_column = st.selectbox("選擇 Target", numeric_targets, index=0)
            feature_frame, skipped_columns = prepare_feature_matrix(df, target_column)
            available_features = list(feature_frame.columns)
            default_features = available_features[: min(20, len(available_features))]
            selected_features = st.multiselect(
                "選擇 Features",
                available_features,
                default=default_features,
                max_selections=39,
            )
            use_kbest = st.checkbox("使用 SelectKBest 進行特徵選擇", value=False)
            if use_kbest and selected_features:
                max_k = min(39, len(selected_features))
                k_features = st.slider("保留特徵數", min_value=14, max_value=max_k, value=min(14, max_k))
            else:
                k_features = None
        else:
            target_column = None
            feature_frame = pd.DataFrame()
            available_features = []
            selected_features = []
            skipped_columns = []

        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("隨機種子", min_value=0, max_value=99999, value=42)

if df is None:
    st.info("請上傳 CSV，或將資料放到 `data/train.csv`。")
    st.markdown(
        """
        **這個工具適合**
        - 連續數值目標預測（例如房價、溫度、銷量、機率）
        - 自動處理可轉換為數值的日期欄位
        - 將低類別數文字欄位轉成虛擬變數（one-hot）

        **不適合直接套用線性回歸的情況**
        - 目標是類別型資料（例如 Yes/No、A/B/C）
        - 文字欄位類別過多且不適合直接編碼
        - 目標值並非連續數值
        """
    )
    st.stop()

if not numeric_targets:
    st.error("資料中沒有可用的連續數值目標欄位，無法直接進行線性回歸。")
    st.write("目前欄位：", list(df.columns))
    st.stop()

if not available_features:
    st.error("資料中沒有可用的輸入特徵，請確認 CSV 內至少有一些數值欄位或可分組文字欄位。")
    st.write("目前欄位：", list(df.columns))
    st.stop()

if len(selected_features) < 14 or len(selected_features) > 39:
    st.warning(f"目前選取 {len(selected_features)} 個特徵；請選擇 14–39 個輸入特徵。")
    st.stop()

st.success(f"資料來源：{source_label}｜{len(df):,} 筆資料｜Target：`{target_column}`｜可用特徵：{len(available_features)}")
if skipped_columns:
    st.caption(f"自動忽略/跳過的欄位：{', '.join(skipped_columns[:10])}。這些欄位可能是識別碼、過多類別文字資料，或不適合直接做回歸。")

with st.expander("查看資料預覽與缺失值", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    preview_cols = [target_column] + list(available_features[:10]) if target_column else []
    if preview_cols:
        missing = df[preview_cols].isna().sum().sort_values(ascending=False)
        st.write("缺失值統計（示範）")
        st.dataframe(missing.rename("missing_count").to_frame(), use_container_width=True)

st.subheader("1. 選擇輸入特徵")
st.write("此工具會自動把日期轉為序數值、將低類別數的文字欄位轉成虛擬變數。請選擇 14–39 個適合的特徵。")

model_df = pd.concat([feature_frame[selected_features], df[target_column].rename(target_column)], axis=1)
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
if len(model_df) < 30:
    st.error("清理缺失值後可用資料少於 30 筆，無法可靠進行訓練與診斷。")
    st.stop()

X = model_df[selected_features]
y = model_df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=float(test_size),
    random_state=int(random_state),
)

if use_kbest:
    selector = SelectKBest(score_func=f_regression, k=min(int(k_features), len(selected_features)))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    final_features = list(X_train.columns[selector.get_support()])
    X_train_fit = X_train[final_features]
    X_test_fit = X_test[final_features]
else:
    final_features = selected_features
    X_train_fit = X_train[final_features]
    X_test_fit = X_test[final_features]

model = LinearRegression()
model.fit(X_train_fit, y_train)
pred = model.predict(X_test_fit)

mse = mean_squared_error(y_test, pred)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(final_features) - 1) if len(y_test) > len(final_features) + 1 else np.nan

st.subheader("2. 模型效能（測試集）")
metric_cols = st.columns(5)
metric_cols[0].metric("MSE", f"{mse:.6f}")
metric_cols[1].metric("RMSE", f"{rmse:.6f}")
metric_cols[2].metric("R²", f"{r2:.6f}")
metric_cols[3].metric("Adjusted R²", f"{adj_r2:.6f}" if np.isfinite(adj_r2) else "N/A")
metric_cols[4].metric("MAE", f"{mae:.6f}")
st.caption(f"訓練集：{len(X_train):,} 筆｜測試集：{len(X_test):,} 筆｜實際模型使用 {len(final_features)} 個特徵")

left, right = st.columns(2)
with left:
    st.markdown("**實際值 vs. 預測值**")
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.35, s=12)
    low = min(float(y_test.min()), float(np.min(pred)))
    high = max(float(y_test.max()), float(np.max(pred)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.markdown("**殘差分布**")
    residuals = y_test.to_numpy() - pred
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=40, edgecolor="white")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Histogram")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.subheader("3. 特徵選擇與係數")
coef_df = pd.DataFrame({"Feature": final_features, "Coefficient": model.coef_})
coef_df["AbsCoefficient"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("AbsCoefficient", ascending=False)
st.dataframe(coef_df.drop(columns="AbsCoefficient"), use_container_width=True)
fig, ax = plt.subplots(figsize=(9, max(4, len(coef_df) * 0.28)))
plot_df = coef_df.sort_values("Coefficient")
ax.barh(plot_df["Feature"], plot_df["Coefficient"])
ax.set_xlabel("Linear regression coefficient")
ax.set_title("Feature Coefficients")
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("4. 五大回歸假設診斷")
st.caption("以下檢定使用訓練集的 OLS 模型。p 值只提供診斷線索，並非正式統計檢定報表。")
X_ols = sm.add_constant(X_train_fit, has_constant="add")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ols = sm.OLS(y_train, X_ols).fit()
    reset_p = float(linear_reset(ols, power=2, use_f=True).pvalue)
    _, bp_lm_p, _, _ = het_breuschpagan(ols.resid, ols.model.exog)
    _, jb_p = jarque_bera(ols.resid)
    dw = float(durbin_watson(ols.resid))
    vif_X = X_train_fit
    if len(vif_X) > 10000:
        vif_X = vif_X.sample(10000, random_state=int(random_state))
    vif_design = sm.add_constant(vif_X, has_constant="add")
    vif_rows = []
    for idx, name in enumerate(vif_design.columns):
        if name == "const":
            continue
        try:
            vif_rows.append((name, float(variance_inflation_factor(vif_design.to_numpy(), idx))))
        except Exception:
            vif_rows.append((name, np.nan))
    max_vif = max((v for _, v in vif_rows if np.isfinite(v)), default=np.nan)

diagnostics = [
    ("線性關係（Ramsey RESET）", reset_p, "p ≥ 0.05：未發現明顯函數形式錯置證據", "p < 0.05：可能存在非線性或模型形式錯置"),
    ("殘差常態性（Jarque–Bera）", float(jb_p), "p ≥ 0.05：未拒絕殘差常態性", "p < 0.05：殘差偏離常態分布"),
    ("同質變異（Breusch–Pagan）", float(bp_lm_p), "p ≥ 0.05：未發現明顯異質變異證據", "p < 0.05：可能存在異質變異"),
]
for name, p, good, bad in diagnostics:
    st.write(f"**{name}** — p-value = `{p:.6g}`")
    st.caption(good if p >= 0.05 else bad)

st.write(f"**殘差獨立性（Durbin–Watson）** — statistic = `{dw:.4f}`")
st.caption("約 2 通常表示沒有明顯的一階自相關；若資料沒有時間／排序結構，DW 的解讀需審慎。")
st.write(f"**多重共線性（VIF）** — 最大 VIF = `{max_vif:.4f}`")
st.caption("VIF > 5 常被視為需要留意；VIF > 10 通常代表較嚴重的共線性。")
with st.expander("查看每個特徵的 VIF"):
    st.dataframe(pd.DataFrame(vif_rows, columns=["Feature", "VIF"]).sort_values("VIF", ascending=False), use_container_width=True)

st.subheader("5. 下載分析結果")
pred_df = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": pred, "Residual": y_test.to_numpy() - pred})
st.download_button(
    "下載測試集預測 CSV",
    pred_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="regression_predictions.csv",
    mime="text/csv",
)
st.download_button(
    "下載模型係數 CSV",
    coef_df.drop(columns="AbsCoefficient").to_csv(index=False).encode("utf-8-sig"),
    file_name="regression_coefficients.csv",
    mime="text/csv",
)

with st.expander("模型摘要（statsmodels OLS）"):
    st.text(ols.summary().as_text())

st.divider()
st.caption("課堂研究用途。若目標是類別型資料，建議改用分類模型而不是一般線性回歸。")
