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
