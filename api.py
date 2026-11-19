
from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'success': True,
        'message': 'Communicare SARIMAX API is running.'
    })


@app.route('/forecast', methods=['POST'])
def forecast():

    try:
        # Receive JSON from PHP
        payload = request.get_json(silent=True)

        if not payload:
            return jsonify({
                'success': False,
                'message': 'Invalid or missing JSON payload.'
            }), 400

        product_code = payload.get('product_code')
        data = payload.get('data', [])
        steps = int(payload.get('steps', 6))

        if not product_code:
            return jsonify({
                'success': False,
                'message': 'Product code is required.'
            }), 400

        if not isinstance(data, list) or len(data) < 24:
            return jsonify({
                'success': False,
                'message': 'At least 24 monthly records are required.'
            }), 400

        if steps < 1 or steps > 24:
            return jsonify({
                'success': False,
                'message': 'Forecast steps must be between 1 and 24.'
            }), 400

        # Convert received data into DataFrame
        df = pd.DataFrame(data)

        if not {'date', 'served'}.issubset(df.columns):
            return jsonify({
                'success': False,
                'message': 'Each record must contain date and served.'
            }), 400

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['served'] = pd.to_numeric(df['served'], errors='coerce')

        if df['date'].isna().any() or df['served'].isna().any():
            return jsonify({
                'success': False,
                'message': 'Invalid date or served quantity.'
            }), 400

        if (df['served'] < 0).any():
            return jsonify({
                'success': False,
                'message': 'Served quantity cannot be negative.'
            }), 400

        # Aggregate records by month
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()

        df = df.groupby('date', as_index=False)['served'].sum()
        df = df.sort_values('date')

        # Create continuous monthly date index
        df = df.set_index('date')

        series = df['served'].asfreq('MS')

        # Fill missing months with zero
        # Only appropriate if missing months mean zero dispensing.
        series = series.fillna(0)

        if len(series) < 24:
            return jsonify({
                'success': False,
                'message': 'At least 24 continuous months are required.'
            }), 400

        # SARIMAX model
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = model.fit(disp=False)

        # Generate forecast
        forecast_result = results.get_forecast(steps=steps)
        forecast_df = forecast_result.summary_frame()

        # Prepare forecast response
        forecast_data = []

        for date, row in forecast_df.iterrows():

            forecast_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_demand': round(max(0, float(row['mean'])), 2),
                'lower_limit': round(max(0, float(row['mean_ci_lower'])), 2),
                'upper_limit': round(max(0, float(row['mean_ci_upper'])), 2)
            })

        return jsonify({
            'success': True,
            'product_code': product_code,
            'historical_records': len(series),
            'forecast_periods': steps,
            'forecast': forecast_data
        })

    except Exception as e:

        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False
    )
