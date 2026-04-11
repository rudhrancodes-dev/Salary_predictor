# Salary Predictor

A machine learning salary prediction project with a Streamlit frontend that estimates salaries from professional and company-related input features.

The application loads a trained model together with a preprocessing pipeline and feature selector, then returns an estimated salary directly in the browser.

## Overview

This project is designed as a lightweight applied ML demo for salary estimation. It combines:

- a trained regression model
- preprocessing and feature selection artifacts
- an interactive Streamlit interface

The user enters a small set of profile and company details, and the system returns a predicted salary value in USD.

## Current Input Features

The app currently takes:

- work year
- experience level
- job title
- company location
- company size

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Interface | Streamlit |
| ML Tooling | scikit-learn, joblib |
| Data Handling | pandas, NumPy |

## Project Structure

```text
Salary_predictor/
├── app.py                          # Streamlit app entry point
├── rf_model_cbrt_tuned.joblib      # Trained regression model
├── feature_selector_cbrt.joblib    # Feature selection artifact
├── preprocessor_cbrt.joblib        # Input preprocessing pipeline
└── README.md
```

## How It Works

The application follows this flow:

1. Load the saved model, selector, and preprocessing pipeline
2. Collect user input from the Streamlit interface
3. Transform the input through the preprocessing pipeline
4. Reduce features through the saved selector
5. Generate a prediction from the trained model
6. Reverse the cube-root transformation used during training
7. Display the estimated salary in USD

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/rudhrancodes-dev/Salary_predictor.git
cd Salary_predictor
```

### 2. Install dependencies

```bash
pip install streamlit joblib numpy pandas scikit-learn
```

### 3. Start the application

```bash
streamlit run app.py
```

## Notes on the Model

The current interface surfaces the model as a demo prediction tool and includes a visible note about approximate accuracy. Based on the app output, the current demo reports:

- approximate MAE of around `$45,366`
- R² of around `0.2777`

This means the project is useful as a demonstration of the workflow, but not yet positioned as a production-grade salary intelligence system.

## What This Repository Demonstrates

- applied machine learning deployment in a simple interface
- use of serialized preprocessing and model artifacts
- structured inference workflow for tabular prediction tasks
- lightweight productization of an ML model with Streamlit

## Possible Improvements

Strong next steps for this project would be:

- documenting the training workflow in more detail
- adding dataset description and evaluation methodology
- expanding the supported job titles and countries
- improving model accuracy with broader data and feature engineering
- deploying the app publicly for easier access

## Author

Built by [Rudhran B](https://github.com/rudhrancodes-dev)
