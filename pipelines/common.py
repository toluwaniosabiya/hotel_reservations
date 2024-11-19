import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import category_encoders as ce


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Reads a parquet file from a datapath and returns a
    dataframe object
    """
    dataset = pq.read_table(data_path)
    data = dataset.to_pydict()
    return pd.DataFrame(data)


def distinguish_label_and_features(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=["bookingLabel"])
    y = df["bookingLabel"]

    return (X, y)


class HotelBooking:
    def __init__(self) -> None:
        self.df = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df
        self._wrangle_numeric_column(self.df)
        self._drop_missing_values(self.df)
        self._create_city_and_country(self.df)
        self._perform_date_calculations(self.df)
        self._calculate_cost_per_day(self.df)
        self._encode_boolean_variables(self.df)
        self._encode_categorical_variables(self.df)
        self._drop_redundant_variables(self.df)

        return self.df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df
        self._wrangle_numeric_column(self.df)
        self._drop_missing_values(self.df)
        self._create_city_and_country(self.df)
        self._perform_date_calculations(self.df)
        self._calculate_cost_per_day(self.df)
        self._encode_boolean_variables(self.df)
        for column in self.categorical_columns:
            self.df[f"{column}_encoded"] = self.df[column].map(
                self.encoding_map[column]
            )
            self.df[f"{column}_encoded"] = self.df[f"{column}_encoded"].fillna(
                self.df[f"{column}_encoded"].mean()
            )
        self._drop_redundant_variables(self.df)

        return self.df

    def _wrangle_numeric_column(self, df: pd.DataFrame):
        # Take absolute value of columns to deal with negatives
        df["starLevel"] = abs(df["starLevel"])
        df["customerReviewScore"] = abs(df["customerReviewScore"])
        df["reviewCount"] = abs(df["reviewCount"])

        # Ensure no review score is greater than 10
        df.loc[df["customerReviewScore"] > 10, "customerReviewScore"] = 10

        self.df = df

    def _drop_missing_values(self, df: pd.DataFrame):
        # Drop rows with missing destinationName values
        df = df.dropna(subset="destinationName")

        self.df = df

    def _create_city_and_country(self, df: pd.DataFrame):
        df["city"] = df["destinationName"].str.split(",").str[0]
        df["country"] = df["destinationName"].str.split(",").str[-1]
        self.df = df

    def __calculate_length_of_stay(self, row):
        if row["checkInDate"] > row["checkOutDate"]:
            return row["checkInDate"] - row["checkOutDate"]
        else:
            return row["checkOutDate"] - row["checkInDate"]

    def __has_weekend(self, row):
        date_range = pd.date_range(start=row["checkInDate"], end=row["checkOutDate"])

        # Check if any day in the range is Friday, Saturday, or Sunday
        return any(date_range.dayofweek >= 4)

    def _perform_date_calculations(self, df: pd.DataFrame):
        # Create lengthOfStay and daysBeforeCheckIn variables
        df["lengthOfStay"] = df.apply(self.__calculate_length_of_stay, axis=1)
        df["lengthOfStay"] = df["lengthOfStay"] / np.timedelta64(
            1, "D"
        )  # This converts timedelta dtype to float

        df["daysBeforeCheckIn"] = df["checkInDate"] - df["searchDate"]
        df["daysBeforeCheckIn"] = df["daysBeforeCheckIn"] / np.timedelta64(
            1, "D"
        )  # This converts timedelta dtype to float

        df["includesWeekend"] = df.apply(self.__has_weekend, axis=1)

        self.df = df

    def _calculate_cost_per_day(self, df: pd.DataFrame):
        df["costPerDay"] = df["numRooms"] * df["minPrice"]

        self.df = df

    def _encode_boolean_variables(self, df: pd.DataFrame):
        # Encode booleans to 1s and 0s
        df[df.select_dtypes(include="bool").columns] = df.select_dtypes(
            include="bool"
        ).astype(int)

        self.df = df

    def _encode_categorical_variables(self, df: pd.DataFrame):
        # Perform target encoding for categorical variables
        self.categorical_columns = [
            "vipTier",
            "rank",
            "starLevel",
            "city",
            "country",
        ]
        target_encoder = ce.TargetEncoder(cols=self.categorical_columns)
        df_encoded = target_encoder.fit_transform(
            df[self.categorical_columns], df["bookingLabel"]
        )
        df = pd.concat([df, df_encoded.add_suffix("_encoded")], axis=1)

        self.df = df
        self.encoding_map = self.__map_encoding(self.categorical_columns)

    def _drop_redundant_variables(self, df: pd.DataFrame):
        df = df.drop(
            columns=[
                "searchId",
                "destinationName",
                "userId",
                "deviceCode",
                "signedInFlag",
                "hotelId",
                "brandId",
                "minStrikePrice",
                "clickLabel",
                "checkInDate",
                "checkOutDate",
                "searchDate",
                "vipTier",
                "rank",
                "starLevel",
                "city",
                "country",
            ]
        )

        self.df = df

    def __map_encoding(self, column_list: list):
        """
        Create a dict of dict to map target encoded values
        This is useful for transforming test/production data
        using already fitted encoding from training.
        """
        mapping_dict = {}
        for column in column_list:
            column1, column2 = column, f"{column}_encoded"
            map_pair_df = self.df[[column1, column2]].set_index(column1)
            map_pair_dict = map_pair_df.to_dict()[column2]
            mapping_dict[column1] = map_pair_dict

        return mapping_dict
