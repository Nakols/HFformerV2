"""Custom formatting functions for orderbook dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import data_formatters.utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class OrderbookFormatter(GenericDataFormatter):
  """Defines and formats data for the volatility dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('symbol', DataTypes.CATEGORICAL, InputTypes.ID),
      ('datetime', DataTypes.DATE, InputTypes.TIME),
    #   ('price', DataTypes.REAL_VALUED, InputTypes.TARGET),  
    #   ('log_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('quantity', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('dollarvol_bid1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('dollarvol_bid3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('dollarvol_bid2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('dollarvol_ask1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('dollarvol_ask2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('dollarvol_ask3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('target', DataTypes.REAL_VALUED, InputTypes.TARGET),  
      ('bid1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid7', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty7', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid8', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty8', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid9', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty9', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bid10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('bidqty10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty4', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask7', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty7', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask8', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty8', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask9', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty9', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ask10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('askqty10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('log_lag26_40_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('log_lag26_50_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('log_lag26_60_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('log_lag26_70_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('log_lag10_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('log_lag20_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('log_lag30_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('log_lag40_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('log_lag50_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('log_lag60_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('log_lag70_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('log_lag80_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    #   ('log_lag90_returns', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),   
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self, df):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """
    print('Formatting train-valid-test splits.')

    len_df = len(df)
    
    len_train = int(0.8*len_df)
    len_valid = int(0.195*len_df)
    
    train = df[0:len_train]
    valid = df[len_train:len_train+len_valid]
    test = df[len_train+len_valid:]

    self.set_scalers(train)
    return (self.transform_inputs(data) for data in [train, valid, test])
    # return (train, valid, test)

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = data_formatters.utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = data_formatters.utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = data_formatters.utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
    self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
        df[[target_column]].values)  # used for predictions

    # Format categorical scalers
    categorical_inputs = data_formatters.utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    real_inputs = data_formatters.utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = data_formatters.utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = self._target_scaler.inverse_transform(predictions[col])

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 125,
        'num_encoder_steps': 100,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5,
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.25,
        'hidden_layer_size': 128,
        'learning_rate': 0.01,
        'minibatch_size': 128,
        'max_gradient_norm': 0.01,
        'num_heads': 8,
        'stack_size': 1
    }

    return model_params
