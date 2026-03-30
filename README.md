# Student Pass/Fail Prediction System

A machine learning-based web application designed to predict student academic outcomes based on various input parameters. The system leverages supervised learning techniques to classify whether a student is likely to pass or fail.

## Overview

This project aims to assist educators and institutions in identifying at-risk students at an early stage, enabling timely intervention and support. The application provides a simple and intuitive web interface for real-time predictions.

## Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn
- **Frontend:** HTML5, CSS3
- **Dataset:** CSV (custom student data)

## Project Structure
- `app.py` — Flask web application
- `model.py` — ML model training and prediction
- `dataset.csv` — Student dataset
- `templates/index.html` — Frontend interface
- `static/style.css` — Stylesheet

## Installation and Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/archirajput/Student-pass-fail.git
cd Student-pass-fail
pip install -r requirements.txt
python app.py
