# Loan Default Prediction

This project implements a loan default prediction system using a dataset of borrower and loan features to identify high-risk applicants using machine learning. The goal is to help lenders reduce default rates and improve decision-making.

## Features

- **Data Preprocessing**: Handles missing values with appropriate imputation and scales features using `StandardScaler`.
- **Class Balancing**: Applies SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance between defaulters and non-defaulters.
- **Model Training**: Trains a LightGBM classifier for accurate and efficient classification.
- **Evaluation Metrics**: Uses Accuracy, Precision, Recall, and F1 Score to assess model performance.
- **Performance Report**: Generates a comprehensive performance report and lender-specific recommendations.

## Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn (preprocessing, metrics)
- Imbalanced-learn (SMOTE)
- LightGBM
- Matplotlib, Seaborn (optional for visualization)

## Final Results

### Model Performance:
- **LightGBM Classifier**:
  - **Class 0 (Non-defaulters)**:
    - Precision: 0.80
    - Recall: 1.00
    - F1 Score: 0.89
  - **Class 1 (Defaulters)**:
    - Precision: 0.25
    - Recall: 0.01
    - F1 Score: 0.01
  - **Overall Accuracy**: 80%


### Key Insight:
The model performs well for non-defaulters but poorly identifies actual defaulters, indicating:
- Class imbalance significantly impacts recall for defaulters.
- SMOTE was applied but may require fine-tuning or additional techniques such as ensemble methods, feature engineering, or threshold adjustments.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-default-prediction.git
   cd loan-default-prediction
   
2. Install required libraries
3. Run the notebook:
- Open Task08.ipynb in Jupyter Notebook and run all cells.

## Output
- Classification report including precision, recall, and F1 scores.
- Detailed recommendations for lenders to improve decision accuracy.


## Contributing
Contributions are welcome! Fork the repo and submit a pull request for improvements, additional models, or visualizations.

## License
This project is open-source and available under the MIT License.
