import sqlite3
import mlflow
import joblib

import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime, timezone
from mlflow.pyfunc import PythonModelContext


class Model(mlflow.pyfunc.PythonModel):
    """
    A custom class to serve the MLFlow in the training pipeline.

    This class acts as a wrapper around the model generated in the training
    pipeline to be registered in the MLFlow model registry. When the model
    is invoked, this class gets the model together with its artifacts and
    other dependencies, like any transformation functions, to ensure the
    model runs properly and makes predictions.

    The class also optionally tracks the input data and model predictions
    for monitoring and feedback.
    """

    def __init__(
        self,
        data_collection_uri: str | None = "reservations-production-table.db",
        data_capture: bool = False,
    ) -> None:

        self.data_collection_uri = data_collection_uri
        self.data_capture = data_capture

    def load_context(self, context: PythonModelContext) -> None:
        """
        Load the model and its artifacts
        """

        # Import the model and its artifacts
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(context.artifacts["model"])
        self.model = xgb_model

        self.data_transformer = joblib.load(context.artifacts["data_transformer"])

    def process_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using the same transformation logic during model training.

        Receives a dataframe and transforms it into data the model can predict on
        """

        try:
            processed_data = self.data_transformer.transform(input_data)
        except Exception:
            print("There was an issues processing the data you passed in!")
            return None

        return processed_data

    def predict(self, input_data: pd.DataFrame | dict) -> pd.DataFrame:
        """
        Make predictions on inputed data and return the predictions.

        Predictions come as a dataframe containing the hotelID,
        predicted class, and prediction probability.
        """

        if not isinstance(input_data, pd.DataFrame | dict):
            print("You have passed in the wrong data format!")
            print("Data needs to be a dataframe or a dictionary.")
            return None

        if isinstance(input_data, dict):
            self.input_data = pd.DataFrame(input_data)

        transformed_input_data = self.process_input(self.input_data)

        if transformed_input_data is not None:
            pred = self.model.predict(transformed_input_data)
            pred_proba = self.model.predict_proba(transformed_input_data)
            # Get the prediction probability for just the positive class
            pred_proba = pred_proba[:, 1]

            self.result = self.serve_output(pred, pred_proba)

        # Define if input data will be tracked based on the value of data_capture
        if self.data_capture:
            self.capture(self.input_data, self.result)

        return self.result

    def serve_output(self, pred: np.ndarray, pred_proba: np.ndarray) -> pd.DataFrame:
        """
        Tranform model predictions to human-readable format.

        Returns a dataframe.
        """

        hotel_id = self.input_data["hotelID"]

        result = pd.DataFrame(
            {
                "Hotel ID": hotel_id,
                "Prediction": pred,
                "Prediction Probability": pred_proba,
            }
        )

        return result

    def capture(self, input_data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """
        Capture input data and prediction results in a database.

        This is useful for ensuring model monitoring and providing
        feedback for model improvement.
        """

        connection = None

        try:
            connection = sqlite3.connect(self.data_collection_uri)

            df = input_data.copy()

            if self.result is not None:
                df["Prediction"] = result["Prediction"]
                df["Prediction Probability"] = result["Prediction Probability"]
                df["date"] = datetime.now(timezone.utc)
            else:
                print(
                    "There was an error logging the required information as there are no results"
                )

            df.to_sql("production_table", connection, if_exists="append", index=False)

        except sqlite3.Error:
            print("There was an error connecting to the database")

        finally:
            if connection:
                connection.close()
