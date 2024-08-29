import pandas as pd
from datetime import datetime, timedelta

def parse_tsf(file_path):
    data = []
    with open(file_path, 'r') as file:
        is_data_section = False
        for line in file:
            line = line.strip()
            if line.startswith('@data'):
                is_data_section = True
                continue  # Skip to the next line after @data

            if is_data_section and line:
                parts = line.split(':')
                if len(parts) == 4:  # Ensuring correct line structure
                    series_name = parts[0]
                    state = parts[1]
                    start_timestamp = datetime.strptime(parts[2], "%Y-%m-%d %H-%M-%S")
                    values = list(map(float, parts[3].split(',')))

                    for i, value in enumerate(values):
                        timestamp = start_timestamp + timedelta(minutes=30 * i)
                        data.append({
                            'series_name': series_name,
                            'state': state,
                            'timestamp': timestamp,
                            'value': value
                        })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Function to convert a series to a time series
def convert_to_timeseries(df, series_name):
    # Filter the DataFrame for the specific series
    series_df = df[df['series_name'] == series_name]
    
    # Set the timestamp as the index and keep only the value column
    series_df = series_df[['timestamp', 'value']].set_index('timestamp')
    
    # Ensure the DataFrame is sorted by timestamp
    series_df.sort_index(inplace=True)
    
    return series_df

# Example usage
file_path = 'test/australian_electricity_demand_dataset.tsf'
df = parse_tsf(file_path)[:2000]

# Convert the T1 series to a time series
t1_timeseries = convert_to_timeseries(df, 'T1')


# Example: Plot the demand for a specific state
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Debugging output
    # amount of series_name: 5, from T1 to T5 -> correspond to 5 austrailian states in this dataset
    #print(df.head())  # Display the first few rows of the DataFrame
    #print(df.columns)  # Check if the columns are named correctly
    # Display the time series
    print(t1_timeseries.head())

    if not df.empty:
        state_df = df[df['state'] == 'NSW']
        plt.figure(figsize=(14, 7))
        plt.plot(state_df['timestamp'], state_df['value'])
        plt.title('Electricity Demand in NSW')
        plt.xlabel('Time')
        plt.ylabel('Demand (MW)')
        plt.show()
    else:
        print("DataFrame is empty, please check the file path and parsing logic.")
