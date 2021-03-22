from math import sqrt, pow
from datetime import datetime
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import csv, json, argparse
import pandas as pd
import matplotlib.pyplot as plt

def RMSE(value_list):
    return sqrt(sum([pow(num['y_hat'] - num['y'], 2) for num in value_list]) / len(value_list))

def get_holidays():
    holiday = pd.DataFrame({
        'ds': list(map(lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), json.load(open('holidays.json', 'r', encoding='utf-8')))),
        'holiday': list(map(lambda x: str(x['name']), json.load(open('holidays.json', 'r', encoding='utf-8')))),
    })
    return holiday[holiday['ds'] >= datetime(2021, 1, 1)]

def main(trainingdata_filename):
    min_max_scaler = MinMaxScaler()
    holidays = get_holidays()
    rename_dic = {'日期':'ds', '備轉容量(萬瓩)':'y'}

    # preprocessing
    train_df = pd.read_csv(trainingdata_filename)
    train_df['日期'] = train_df['日期'].apply(lambda x: datetime.strptime(str(x), '%Y/%m/%d'))
    train_df['備轉容量(萬瓩)'] = train_df['備轉容量(萬瓩)'].apply(lambda x: 10 * int(x))
    train_df['備轉容量(萬瓩)'] =  pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(train_df['備轉容量(萬瓩)'])))

    # model_training
    m = Prophet(holidays = holidays)
    m.fit(train_df.rename(rename_dic, axis=1))
    
    # predicting
    future = m.make_future_dataframe(periods=10)
    forecast = m.predict(future)
    
    # inverse normolization
    forecast['yhat'] = pd.DataFrame(min_max_scaler.inverse_transform(pd.DataFrame(forecast['yhat'])))
    forecast['yhat_lower'] = pd.DataFrame(min_max_scaler.inverse_transform(pd.DataFrame(forecast['yhat_lower'])))
    forecast['yhat_upper'] = pd.DataFrame(min_max_scaler.inverse_transform(pd.DataFrame(forecast['yhat_upper'])))
    
    return forecast

def output_csv(output_filename, forecast, start, end):
    start_date = start.split('-')
    end_date = end.split('-')
    f = open(output_filename,'w',newline='')
    w = csv.writer(f)
    w.writerow(['date','operating_reserve(MW)'])
    for i in range(int(start_date[2]), int(end_date[2]) + 1):
        x = forecast[(forecast['ds'] == datetime(2021,3,i))][['ds', 'yhat']]
        date = x['ds'].get(x['ds'].index[0]).strftime("%Y%m%d")
        w.writerow([date,int(x['yhat'])])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='training_data.csv', help='input training data file name')
    parser.add_argument('--output', default='submission.csv', help='output file name')
    args = parser.parse_args()
    
    forecast = main(args.training)
    output_csv(output_filename = args.output, forecast = forecast, start = "2021-3-23", end = "2021-3-29")
    
    # test
    predict_date = [ datetime.strptime(f'2021-03-2{i}', '%Y-%m-%d')  for i in range(3, 10) ]
    y_dict = pd.DataFrame({ # 未來一周台電預測值
        'ds': ['2021-03-23', '2021-03-24', '2021-03-25', '2021-03-26', '2021-03-27', '2021-03-28', '2021-03-29'],
        'y': [3070, 3260, 3160, 3200, 2840, 3090, 3050]
    })
    
    forecast_list = []
    y_list = []
    y_hat_list = []
    
    for date in predict_date:
        y = y_dict[y_dict['ds'] == date.strftime('%Y-%m-%d')]['y'].values.tolist()[0]
        y_hat = forecast[forecast['ds'] == date]['yhat'].values.tolist()[0]
        y_list.append(y)
        y_hat_list.append(y_hat)
        forecast_list.append( {'y': y, 'y_hat': y_hat} )

    # RMSE
    print(f'RMSE = {RMSE(forecast_list)}')
    
    # draw
    plt.plot(predict_date, y_list, color='red', label='real_value')
    plt.plot(predict_date, y_hat_list, color='blue', label='predicted_value')
    plt.legend()
    plt.show()