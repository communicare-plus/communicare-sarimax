
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import hmac
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

warnings.filterwarnings("ignore")


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MIN_RECORDS = 15
FORECAST_STEPS = 3
SEASONAL_PERIOD = 12
MAX_ITERATIONS = 50

# Reduced candidate combinations: 2 x 2 = 4 models
ORDER_CANDIDATES = [
    (1, 1, 1),
    (1, 1, 0)
]

SEASONAL_CANDIDATES = [
    (0, 0, 0, SEASONAL_PERIOD),
    (1, 0, 0, SEASONAL_PERIOD)
]


# --------------------------------------------------
# API KEY VALIDATION
# --------------------------------------------------

def validate_api_key():

    expected_key = os.environ.get('COMMUNICARE_API_KEY')
    received_key = request.headers.get('X-API-KEY', '')

    if not expected_key:
        return jsonify({
            'success': False,
            'message': 'API authentication is not configured.'
        }), 500

    if not hmac.compare_digest(received_key, expected_key):
        return jsonify({
            'success': False,
            'message': 'Unauthorized. Invalid API key.'
        }), 401

    return None


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route('/', methods=['GET'])
def home():

    return jsonify({
        'success': True,
        'message': 'Communicare SARIMAX API is running.'
    })


# --------------------------------------------------
# PREPARE MONTHLY DATA
# --------------------------------------------------

def prepare_monthly_data(data):

    df = pd.DataFrame(data)

    if not {'date', 'served'}.issubset(df.columns):
        raise ValueError(
            'Each record must contain date and served.'
        )

    df['date'] = pd.to_datetime(
        df['date'],
        errors='coerce'
    )

    df['served'] = pd.to_numeric(
        df['served'],
        errors='coerce'
    )

    if df['date'].isna().any() or df['served'].isna().any():
        raise ValueError(
            'Invalid date or served quantity.'
        )

    if (df['served'] < 0).any():
        raise ValueError(
            'Served quantity cannot be negative.'
        )

    # Normalize to monthly dates
    df['date'] = (
        df['date']
        .dt.to_period('M')
        .dt.to_timestamp()
    )

    # Aggregate duplicate months
    df = (
        df.groupby('date', as_index=False)['served']
        .sum()
        .sort_values('date')
    )

    # Fill missing months
    monthly_index = pd.date_range(
        start=df['date'].min(),
        end=df['date'].max(),
        freq='MS'
    )

    df = (
        df.set_index('date')
        .reindex(monthly_index, fill_value=0)
    )

    df.index.name = 'date'

    return df['served'].astype(float)


# --------------------------------------------------
# CALCULATE MAE AND RMSE
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    return float(mae), float(rmse)


# --------------------------------------------------
# FIT SARIMAX MODEL
# --------------------------------------------------

def fit_sarimax(series, order, seasonal_order):

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(
        disp=False,
        maxiter=MAX_ITERATIONS,
        method='lbfgs'
    )

    return results


# --------------------------------------------------
# SARIMAX CANDIDATE EVALUATION
# --------------------------------------------------

def evaluate_candidates(series):

    # Reserve the latest 3 months for testing
    test_size = 3

    if len(series) <= test_size:
        raise ValueError(
            'Not enough records for model evaluation.'
        )

    train_data = series.iloc[:-test_size]
    test_data = series.iloc[-test_size:]

    candidates = []

    for order in ORDER_CANDIDATES:

        for seasonal_order in SEASONAL_CANDIDATES:

            try:

                results = fit_sarimax(
                    train_data,
                    order,
                    seasonal_order
                )

                predictions = results.forecast(
                    steps=test_size
                )

                predictions = np.maximum(
                    np.asarray(predictions, dtype=float),
                    0
                )

                mae, rmse = calculate_metrics(
                    test_data.values,
                    predictions
                )

                aic = float(results.aic)

                if not all(
                    np.isfinite(value)
                    for value in [aic, mae, rmse]
                ):
                    continue

                candidates.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': aic,
                    'mae': mae,
                    'rmse': rmse
                })

            except Exception as e:

                print(
                    f"Candidate failed: {order} "
                    f"{seasonal_order} - {str(e)}"
                )

                continue

    if not candidates:
        raise ValueError(
            'Unable to fit any SARIMAX candidate model.'
        )

    candidates.sort(
        key=lambda x: (x['rmse'], x['mae'])
    )

    best_candidate = candidates[0]

    candidate_results = []

    for candidate in candidates:

        candidate_results.append({

            'candidate_model': (
                f"SARIMAX{candidate['order']}x"
                f"{candidate['seasonal_order']}"
            ),

            'order': list(candidate['order']),

            'seasonal_order': list(
                candidate['seasonal_order']
            ),

            'aic': round(candidate['aic'], 2),

            'mae': round(candidate['mae'], 2),

            'rmse': round(candidate['rmse'], 2),

            'selected': candidate is best_candidate

        })

    return (
        best_candidate,
        candidate_results,
        train_data,
        test_data
    )


# --------------------------------------------------
# FORECAST API
# --------------------------------------------------

@app.route('/forecast', methods=['POST'])
def forecast():

    auth_error = validate_api_key()

    if auth_error:
        return auth_error

    try:

        payload = request.get_json(silent=True)

        if not payload:
            return jsonify({
                'success': False,
                'message': 'Invalid or missing JSON payload.'
            }), 400

        product_code = payload.get('product_code')
        data = payload.get('data', [])

        if not product_code:
            return jsonify({
                'success': False,
                'message': 'Product code is required.'
            }), 400

        if not isinstance(data, list) or len(data) == 0:
            return jsonify({
                'success': False,
                'message': 'Historical data must be a non-empty list.'
            }), 400

        # ------------------------------------------
        # PREPARE HISTORICAL DATA
        # ------------------------------------------

        series = prepare_monthly_data(data)

        if len(series) < MIN_RECORDS:

            return jsonify({

                'success': False,

                'message': (
                    f'At least {MIN_RECORDS} continuous '
                    'monthly records are required.'
                ),

                'available_records': len(series)

            }), 400

        # ------------------------------------------
        # EVALUATE CANDIDATES
        # ------------------------------------------

        (
            best_candidate,
            candidate_results,
            train_data,
            test_data
        ) = evaluate_candidates(series)

        best_order = best_candidate['order']
        best_seasonal_order = best_candidate['seasonal_order']

        # ------------------------------------------
        # FINAL MODEL USING ALL HISTORICAL DATA
        # ------------------------------------------

        final_results = fit_sarimax(
            series,
            best_order,
            best_seasonal_order
        )

        final_aic = float(final_results.aic)

        # ------------------------------------------
        # GENERATE 3-MONTH FORECAST
        # ------------------------------------------

        forecast_result = final_results.get_forecast(
            steps=FORECAST_STEPS
        )

        forecast_df = forecast_result.summary_frame()

        forecast_data = []

        for date, row in forecast_df.iterrows():

            predicted_demand = max(
                0,
                float(row['mean'])
            )

            lower_limit = max(
                0,
                float(row['mean_ci_lower'])
            )

            upper_limit = max(
                0,
                float(row['mean_ci_upper'])
            )

            forecast_data.append({

                'date': date.strftime('%Y-%m-%d'),

                'predicted_demand': round(
                    predicted_demand, 2
                ),

                'lower_limit': round(
                    lower_limit, 2
                ),

                'upper_limit': round(
                    upper_limit, 2
                )

            })

        # ------------------------------------------
        # MODEL EVALUATION
        # ------------------------------------------

        model_evaluation = {

            'aic': round(
                best_candidate['aic'], 2
            ),

            'final_model_aic': round(
                final_aic, 2
            ),

            'mae': round(
                best_candidate['mae'], 2
            ),

            'rmse': round(
                best_candidate['rmse'], 2
            ),

            'selection_criteria': (
                'Lowest RMSE, then lowest MAE'
            ),

            'training_records': len(train_data),

            'testing_records': len(test_data)

        }

        # ------------------------------------------
        # HISTORICAL RECORDS
        # ------------------------------------------

        historical_records = [

            {
                'date': date.strftime('%Y-%m'),
                'served': round(float(value), 2)
            }

            for date, value in series.items()

        ]

        # ------------------------------------------
        # API RESPONSE
        # ------------------------------------------

        return jsonify({

            'success': True,

            'product_code': product_code,

            'candidate_model': (
                f"SARIMAX{best_order}x{best_seasonal_order}"
            ),

            'model_parameters': {

                'order': list(best_order),

                'seasonal_order': list(
                    best_seasonal_order
                )

            },

            'model_evaluation': model_evaluation,

            'candidate_results': candidate_results,

            'historical_records': historical_records,

            'forecast_periods': FORECAST_STEPS,

            'forecast': forecast_data

        }), 200

    except ValueError as e:

        return jsonify({

            'success': False,

            'message': str(e)

        }), 400

    except Exception as e:

        import traceback

        traceback.print_exc()

        return jsonify({

            'success': False,

            'message': 'Forecast processing failed.',

            'error': str(e)

        }), 500


# --------------------------------------------------
# RUN LOCAL SERVER
# --------------------------------------------------

if __name__ == '__main__':

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False
    )
