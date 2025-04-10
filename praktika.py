import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "D:/transactions_data_preprocessed.csv"
chunk_size = 100_000
data = pd.read_csv(file_path, nrows=chunk_size)

if 'zip' in data.columns:
    data['zip'].fillna(data['zip'].mode()[0], inplace=True)
if 'merchant_city' in data.columns:
    data['merchant_city'].fillna('unknown', inplace=True)
if 'merchant_state' in data.columns:
    data['merchant_state'].fillna('unknown', inplace=True)
data.fillna(0, inplace=True)  # Заполняем оставшиеся NaN нулями

def remove_outliers(df, column):
    if column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

columns_to_check = ['amount', 'transaction_count_by_client', 'days_since_last_transaction']
for col in columns_to_check:
    data = remove_outliers(data, col)

label_encoder = LabelEncoder()
if 'use_chip' in data.columns:
    data['use_chip'] = label_encoder.fit_transform(data['use_chip'])
if 'merchant_state' in data.columns:
    data['merchant_state'] = label_encoder.fit_transform(data['merchant_state'])

columns_to_scale = ['amount', 'transaction_count_by_client', 'avg_transaction_by_card',
                    'unique_merchants_by_client', 'same_city_transaction_ratio']

columns_to_scale = [col for col in columns_to_scale if col in data.columns]

scaler = StandardScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
data['anomaly_score'] = iso_forest.fit_predict(data[columns_to_scale])

plt.figure(figsize=(8, 6))
sns.histplot(data, x='amount', kde=True, bins=50, hue=data['anomaly_score'].astype(str))
plt.title("Распределение сумм транзакций с аномалиями")
plt.xlabel("Сумма транзакции")
plt.ylabel("Частота")
plt.show()

print("Количество аномалий:", sum(data['anomaly_score'] == -1))
print("Общий размер датасета:", len(data))
print(data.head())

data.to_csv('transactions_with_anomalies.csv', index=False)
anomalies = data[data['anomaly_score'] == -1]
print(anomalies.head())
anomaly_percentage = len(anomalies) / len(data) * 100
print(f"Процент аномалий: {anomaly_percentage:.2f}%")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='amount', y='transaction_count_by_client', hue='anomaly_score', palette={1: 'blue', -1: 'red'})
plt.title("Аномалии по сумме и количеству транзакций")
plt.show()

