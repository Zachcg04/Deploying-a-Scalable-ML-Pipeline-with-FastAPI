# Model Card

## Model Details
This project uses a supervised machine learning model trained on the Census Income dataset. The model was built using scikit-learn. After testing a few options, I selected a Random Forest Classifier because it balanced accuracy and interpretability well. The model is trained to predict whether an individual earns more than $50K per year based on demographic and work-related features. 

## Intended Use
The model is intended for educational purposes as part of a project to learn about building and deploying scalable ML pipelines with FastAPI. It should not be used for making real-world financial or hiring decisions.

## Training Data
The training dataset comes from the UCI Census Income dataset. It includes features such as age, education, occupation and hours worked per week. The data was preprocessed using one-hot encoding for categorical variables and scaling for numerical values.

## Evaluation Data
The dataset was split into training and test sets. The test set was held out from training to evaluate the model’s performance. I also generated metrics across different slices of the data (for example, by education level) to see how well the model performs on different groups.

## Metrics
On the overall test set, the model achieved the following scores:
- Precision: 0.7419
- Recall: 0.6384
- F1-score: 0.6863

These numbers may vary slightly depending on random seeds and retraining.

## Ethical Considerations
This model is trained on census data, which may include social and demographic biases. Predictions should not be used to make decisions about individuals in the real world. The goal of this project was only to practice model development, testing, and deployment.

## Limitations
- The model only works with the specific dataset it was trained on.  
- It may not generalize well to other populations or newer data.  
- Some slices of the data may have lwer performance (for example, smaller subgroups may have less accurate predictions).  

## Author
Created by Zach Garrett
Date: September 13, 2025
