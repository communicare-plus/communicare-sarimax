
import pandas as pd
from db import get_connection

product_code = 'IT00001552'

print("Connecting to MySQL...")

conn = get_connection()
cursor = conn.cursor(dictionary=True)

try:
    query = """
        SELECT
            b.tran_date AS date,
            SUM(a.qty_served) AS served
        FROM pharmacy_dispensing_details AS a
        INNER JOIN pharmacy_dispensing_masterlist AS b
            ON b.pres_uniq = a.pres_uniq
        WHERE a.product_code = %s
        GROUP BY b.tran_date
        ORDER BY b.tran_date ASC
    """

    print("Fetching pharmacy data...")

    cursor.execute(query, (product_code,))
    rows = cursor.fetchall()

    df = pd.DataFrame(rows)

    if not df.empty:

        df['date'] = pd.to_datetime(df['date'])
        df['served'] = pd.to_numeric(df['served'])

        df = df.set_index('date')

        # Monthly frequency
        df = df.asfreq('MS', fill_value=0)

        # Export CSV
        df.to_csv('pharmacy_data.csv')

        print("\nSuccessfully retrieved pharmacy data!")
        print("Total records:", len(df))
        print(df.tail(10))

        print("\nCSV successfully generated!")

    else:
        print("No records found for:", product_code)

finally:
    cursor.close()
    conn.close()
