import os
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifact/common", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
            }

            params = {
                "Linear Regression": {
                     'fit_intercept': [True, False],
                    'copy_X': [True],
                    'n_jobs': [-1] 
                },
                
            }
            # Evaluate models with cross-validation
            cv_scores = {}
            for name, model in models.items():
                if params[name]:  # If model has hyperparameters
                    grid_search = GridSearchCV(
                        model, params[name], cv=5, scoring='r2', n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    cv_scores[name] = grid_search.best_score_
                    models[name] = grid_search.best_estimator_
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    cv_scores[name] = scores.mean()
                    model.fit(X_train, y_train)

            # Select best model based on cross-validation scores
            best_model_name = max(cv_scores, key=cv_scores.get)
            best_model = models[best_model_name]
            best_score = cv_scores[best_model_name]

            if best_score < 0.6:
                raise CustomException("No suitable model found with an acceptable score.")

            logging.info(f"Selected model: {best_model_name} with CV R² score: {best_score}")

            # Create ensemble of top 3 models
            top_models = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            ensemble_models = [(name, models[name]) for name, _ in top_models]
            
            if len(ensemble_models) > 1:
                ensemble = VotingRegressor(estimators=ensemble_models)
                ensemble.fit(X_train, y_train)
                ensemble_score = r2_score(y_test, ensemble.predict(X_test))
                
                if ensemble_score > best_score:
                    best_model = ensemble
                    best_model_name = "Ensemble"
                    best_score = ensemble_score
                    logging.info(f"Using ensemble model with R² score: {ensemble_score}")

            # Save the best model
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate on test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)

            logging.info(f"Test set metrics:")
            logging.info(f"R² score: {r2_square}")
            logging.info(f"MSE: {mse}")
            logging.info(f"MAE: {mae}")

            print(f"Selected Model: {best_model_name}")
            print(f"R² Score: {r2_square:.4f}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

"""

Linear Regression is good for several reasons, especially in the context of diabetic risk prediction. Let me explain:

Interpretability:
Linear Regression provides clear, interpretable coefficients for each feature
You can easily understand how each factor (like age, BMI, etc.) affects the diabetes risk
This is crucial for medical applications where understanding the relationship between variables is important

Computational Efficiency:
It's much faster to train and predict compared to complex models 
Requires less memory and computational resources
Perfect for real-time predictions in a production environment

Robustness:
Less prone to overfitting compared to complex models
Works well even with smaller datasets
More stable predictions across different data distributions

Medical Domain Suitability:
Many medical relationships are linear or can be approximated linearly
Easier to validate and explain to medical professionals
Can be easily integrated into medical decision support systems

"""
