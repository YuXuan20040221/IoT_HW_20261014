# CSV Regression Explorer

A responsive, static multiple-linear-regression dashboard designed for GitHub Pages. CSV files are parsed and analyzed in the visitor's browser; the app does not upload dataset contents to a server.

## Features

- Import different CSV datasets and inspect detected columns and row counts.
- Select a continuous numeric target and one or more input features.
- Automatically skip identifier/high-cardinality text columns, encode low-cardinality categories, and parse date columns.
- Exclude rows with missing or invalid values in the selected columns.
- Configure the test split, maximum feature count, feature-selection strategy, and random seed. Blank numeric inputs use documented defaults.
- Compare R² and RMSE across feature counts; display R-squared, MAE, MSE, and RMSE on the holdout test set.
- Inspect the fitted equation, coefficients, live predictions, and the largest test-set residuals.
- Download test-set predictions as CSV.
- Deploy as a static GitHub Pages site using GitHub Actions.

## CRISP-DM 流程說明

本專案依循 CRISP-DM（Cross-Industry Standard Process for Data Mining）資料探勘流程，將各階段應用於 CSV 線性回歸分析：

1. **Business Understanding（商業理解）**
   確認分析目的與預測問題，例如使用輸入特徵預測連續數值目標。先判斷問題是否適合以線性回歸處理；若目標是類別，應考慮分類模型。
2. **Data Understanding（資料理解）**
   匯入 CSV 後檢視欄位與資料筆數，辨識數值目標、輸入特徵、日期、類別欄位，以及可能是識別碼或不適合分析的欄位。
3. **Data Preparation（資料準備）**
   將日期欄位轉為數值、將低類別數的文字欄位轉為虛擬變數，略過識別碼與高類別數文字欄位，並排除所選目標或特徵有缺漏、格式錯誤或非有限值的資料列。
4. **Modeling（建模）**
   以訓練資料建立多元線性回歸模型；可設定特徵選擇方式、最多使用的特徵數、訓練／測試資料比例及隨機種子。特徵數比較與選擇使用訓練資料內的驗證切分，不以測試資料挑選模型。
5. **Evaluation（評估）**
   使用保留的測試資料評估模型，顯示 R²、MAE、MSE、RMSE，並提供不同特徵數的效能比較、回歸方程式、係數、即時預測及測試集殘差檢視。
6. **Deployment（部署）**
   透過 GitHub Actions 執行測試並部署靜態網站至 GitHub Pages。使用者上傳的 CSV 在瀏覽器內分析，不會隨部署傳送至伺服器。

## Run locally

No build step or server is required. Open `index.html` in a browser, or serve the repository with any static file server. Papa Parse is loaded from jsDelivr to read CSV files.

The regression calculation and tests use Node.js 20 or newer:

```sh
node --test tests/regression.test.mjs
```

## Deploy to GitHub Pages

The workflow at `.github/workflows/pages.yml` runs the regression tests on pushes to `main`, pull requests, and manual dispatch. On pushes to `main` (and manual dispatch), it packages the static site and deploys it to GitHub Pages.

In the GitHub repository, open **Settings → Pages** and set the build/deployment source to **GitHub Actions**. Push to `main`; the published URL appears in the workflow's deployment environment.

## Data handling and limitations

- CSV processing is performed locally in the browser. Papa Parse is fetched from a public CDN; no CSV data is sent to that CDN.
- Files larger than 50 MB are rejected. Up to 50,000 data rows are analyzed; additional rows are excluded with an on-page notice.
- Linear regression requires a continuous numeric target and enough complete observations. Binary/category targets should use a classification model instead.
- Constant, repeated, or highly collinear predictors may not be suitable. The app reports model-fitting errors rather than silently substituting a different method.
- R² and error measures are holdout-set performance estimates, not evidence of causality or a guarantee of future performance.
- `app.py` and `requirements.txt` are retained from the original Streamlit project; the GitHub Pages workflow publishes the static HTML, CSS, and JavaScript app instead.
