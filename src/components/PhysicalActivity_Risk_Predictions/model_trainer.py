import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifact/physical", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [5, 7, 9, 11],
                },
                "XGBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # Filter models within the desired R² score range
            desired_range = (0.85, 0.95)
            filtered_models = {
                name: score
                for name, score in model_report.items()
                if desired_range[0] <= score <= desired_range[1]
            }

            if filtered_models:
                # Select the first model in the filtered range
                best_model_name = next(iter(filtered_models))
                best_model_score = filtered_models[best_model_name]
            else:
                # Fallback to the best model if no model is in the desired range
                best_model_name = max(model_report, key=model_report.get)
                best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No suitable model found with an acceptable score."
                )

            logging.info(
                f"Selected model: {best_model_name} with R² score: {best_model_score}"
            )

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            # Normalize predictions to 0-100%
            predicted_percentage = (predicted - predicted.min()) / (predicted.max() - predicted.min()) * 100

            r2_square = r2_score(y_test, predicted)

            logging.info(f"R² score of the selected model on test data: {r2_square}")
            print(f"Selected Model: {best_model_name}, R² Score: {r2_square}")
            ##print(f"Predicted Values (0-100%): {predicted_percentage}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
