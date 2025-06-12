import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

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
                # Decision Tree is ideal for physical activity risk prediction because:
                # 1. Fast training and prediction
                # 2. Easy to interpret and visualize
                # 3. Works well with SHAP for feature importance
                # 4. Handles both numerical and categorical features
                # 5. Provides clear decision rules
                "Decision Tree": DecisionTreeRegressor(
                    random_state=42,  # For reproducibility
                ),
            }

            params = {
                "Decision Tree": {
                    # Maximum depth of the tree
                    "max_depth": [3, 5, 7, 10],
                    # Minimum samples required to split a node
                    "min_samples_split": [2, 5, 10],
                    # Minimum samples required at leaf node
                    "min_samples_leaf": [1, 2, 4],
                    # Maximum features to consider for best split
                    "max_features": ['sqrt', 'log2', None],
                    # Criterion for measuring quality of split
                    "criterion": ['squared_error', 'absolute_error']
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

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
