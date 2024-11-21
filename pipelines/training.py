import dotenv
import os
import mlflow
import joblib
import tempfile

import xgboost as xgb
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models import infer_signature, infer_pip_requirements
from metaflow import (
    FlowSpec,
    # Parameter,
    step,
    # pypi_base,
    project,
    current,
    card,
)
from common import load_dataset
from inference import Model


@project(name="hotel_reservations")
# @pypi_base()
class Training(FlowSpec):
    """
    Training Pipeline.

    This pipline trains, evaluates and registers a model
    to predict if a user will book an hotel from a list
    or not.
    """

    data_path = list((Path(__file__).parent.parent / "data").glob("*"))[0]

    @card
    @step
    def start(self):
        """
        Start and prepare the training pipeline
        """
        print(self.data_path)

        self.df = load_dataset(self.data_path)
        self.df = self.df[:500]

        dotenv.load_dotenv()
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        try:
            # Start a new mlflow run to track the experiment
            # using the same run_id from metaflow
            # for easy tracking across platforms

            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLFlow server: \
                {self.mlflow_tracking_uri}"
            raise RuntimeError(message) from e

        self.next(self.transform)

    @card
    @step
    def transform(self):
        """
        Preprocess and transform dataset
        """

        # Split into training and test sets based on
        # unique indices
        unique_search_ids = self.df["searchId"].unique()
        self.train_ids, self.test_ids = train_test_split(
            unique_search_ids, test_size=0.15, random_state=42
        )

        from common import HotelBooking

        self.data_transformer = HotelBooking()

        self.df_train = self.data_transformer.fit_transform(
            self.df[self.df["searchId"].isin(self.train_ids)]
        )
        self.df_test = self.data_transformer.transform(
            self.df[self.df["searchId"].isin(self.test_ids)]
        )

        mapper = self.data_transformer.encoding_map
        cat_cols = [key for key in mapper.keys()]

        # The step below ensures the test set doesn't contain any empty columns
        # since the transformation on the test set is based on the fit
        # from the training set.
        for column in cat_cols:
            print(column)
            print(self.df_test[f"{column}_encoded"].isna().sum())

        self.next(self.split)

    @card
    @step
    def split(self):
        """
        Split data into training and test sets
        """
        from common import distinguish_label_and_features

        self.X_train, self.y_train = distinguish_label_and_features(self.df_train)
        self.X_test, self.y_test = distinguish_label_and_features(self.df_test)

        # The print statements below are necessary to ensure all
        # splits have the same shape
        print(self.X_train.columns)
        print(self.X_test.columns)
        print(self.y_train.name)
        print(self.y_test.name)

        self.next(self.train)

    @card
    @step
    def train(self):
        """
        Train model and log the training parameters
        """
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)

            self.scale_pos_weight = sum(self.y_train == 0) / sum(self.y_train == 1)
            # The above controls the balance of positive and negative weights

            xgb_model = xgb.XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
            )

            self.param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 1],
                "min_child_weight": [1, 3, 5],
            }

            # Set up the GridSearchCV
            grid_search_xgb = GridSearchCV(
                estimator=xgb_model,
                param_grid=self.param_grid,
                scoring="recall",  # Recall is chosen because correctly identifying the positive class is desirable
                cv=5,
                verbose=2,
                n_jobs=-1,
            )

            grid_search_xgb.fit(self.X_train, self.y_train)
            print("Best Parameters:", grid_search_xgb.best_params_)
            self.model = grid_search_xgb.best_estimator_

        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        """
        Evaluate and log metrics
        """
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        # Create a DataFrame with the true and predicted values
        results_df = pd.DataFrame(
            {
                "True_Label": self.y_test,
                "Predicted_Label": self.y_pred,
                "Prediction_probaility": self.y_pred_proba,
            }
        )

        # Save the DataFrame to a CSV file
        results_df.to_csv("results.csv", index=False)

        # Get a classification report.
        self.report = classification_report(
            self.y_test,
            self.y_pred,
            target_names=["Not Booked", "Booked"],
            output_dict=True,
        )
        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)

        self.pos_class_recall = round(self.report["Booked"]["recall"], 2)
        self.mean_recall = round(
            np.mean(
                (self.report["Booked"]["recall"], self.report["Not Booked"]["recall"])
            ),
            2,
        )

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "pos_class_recall": self.pos_class_recall,
                    "mean_recall": self.mean_recall,
                }
            )

        self.next(self.register_model)

    @card
    @step
    def register_model(self):
        """
        Registers the trained model in MLFlow.

        Wraps the trained model in the Model class from the inference file.
        Also includes model artifact to be logged to MLFlow.
        """

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            tempfile.TemporaryDirectory() as directory,
        ):
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=Model(data_capture=True),
                registered_model_name="hotel-reservations",
                code_paths=[
                    (Path(__file__).parent / "inference.py").as_posix(),
                    (Path(__file__).parent / "common.py").as_posix(),
                ],
                artifacts=self._get_model_artifacts(directory),
                signature=self._get_model_signature(),
                pip_requirements=self._get_model_pip_requirements(),
            )

        self.next(self.end)

    @card
    @step
    def end(self):
        """End the pipeline"""

    def _get_model_artifacts(self, directory: str):
        """
        Return the list of artifacts that will be included in the model.

        This will include the functions to transform raw inputs into a
        format acceptable by the model.
        """
        # Save the model
        model_path = (Path(directory) / "hotel_reservations.json").as_posix()
        self.model.save_model(model_path)

        # Save the transformer function
        transformer_path = (Path(directory) / "transformer.joblib").as_posix()
        joblib.dump(self.data_transformer, transformer_path)

        return {
            "model": model_path,
            "data_transformer": transformer_path,
        }

    def _get_model_signature(self):
        """
        Return the model's signature

        This will include the expected format for model inputs and outputs.
        It gives some information about the correct use of the model.
        """
        # input_dict = self.df[:1].drop(columns="bookingLabel").transpose().to_dict()[0]
        model_input = (
            self.df[self.df["searchId"].isin(self.test_ids)][:1]
            .drop(columns="bookingLabel")
            .to_dict(orient="records")[0]
        )

        model_output = {
            "Hotel ID": model_input["hotelId"],
            "Prediction": self.y_pred[0],
            "Prediction Probability": self.y_pred_proba[0],
        }

        return infer_signature(
            model_input=model_input,
            model_output=model_output,
        )

    def _get_model_pip_requirements(self):
        """
        Return list of required packages to run model
        """

        with open(Path(__file__).parent.parent / "requirements.txt", "r") as file:
            lines = file.readlines()

            # Remove comments and blank lines, and strip whitespace
            requirements = [line.strip() for line in lines]

        requirements = [item for item in requirements if not item.startswith("#")]

        return requirements


if __name__ == "__main__":
    Training()
