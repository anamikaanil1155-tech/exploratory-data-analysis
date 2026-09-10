# Exploratory Data Analysis (EDA) Project

## Project Goal
Analyze a tabular dataset to discover patterns, distributions, relationships, correlations, outliers, and important factors.

This project is designed to be easy to run in **Google Colab** or **Jupyter Notebook** and easy to upload to **GitHub**.

## What this project covers
1. Data Loading & Initial Inspection
2. Summary Statistics
3. Missing Data & Duplicate Handling
4. Univariate Analysis
5. Bivariate & Multivariate Analysis
6. Insights Extraction & Structured Report

## Dataset used for the ready-to-run example
The notebook uses the **Titanic dataset** from Seaborn by default. You can later replace it with your own CSV file.

## Files
- `EDA_Project.ipynb` — complete step-by-step notebook
- `eda_analysis.py` — Python script version
- `report_template.md` — final report template
- `requirements.txt` — required Python libraries
- `README.md` — project instructions

## How to run in Google Colab
1. Open Google Colab.
2. Upload `EDA_Project.ipynb`.
3. Run each cell from top to bottom.
4. The notebook automatically loads the Titanic dataset.
5. Read the printed results and charts.
6. Use the generated findings to complete `report_template.md`.

## How to use your own CSV
In the notebook, change:
```python
DATA_SOURCE = "titanic"
```
to:
```python
DATA_SOURCE = "csv"
CSV_PATH = "data/your_dataset.csv"
```

Then upload your CSV into the `data` folder.

## GitHub
Upload these files to a GitHub repository. The `.ipynb` notebook can be opened directly in GitHub, and you can use the **Open in Colab** option after connecting the repository to Google Colab.

## Important
Correlation shows how two numerical variables move together. It does **not** prove that one variable causes the other.
