# 2021-DSAI-HW1作業說明

## 使用說明
安裝相依套件
```
pip install -r requirements.txt
```
運行主程式
```
python app.py --training training_data.csv --output submission.csv
```


## 資料說明

### 台灣電力公司_本年度每日尖峰備轉容量率
#### 資料來源
https://data.gov.tw/dataset/25850
#### 時間範圍
2021年1月1日 ~ 2021年3月21日


### 台灣周末假期與國定假日資料
#### 資料來源
https://www.dgpa.gov.tw/information?uid=83&pid=10173
#### 時間範圍
2021年1月1日 ~ 2021年3月21日

## 模型說明

### fbprophet

#### 模型介紹
時間序列模型，主要藉由多尺度的週期性，例如:一周，一個月.....等，與重要假期資料來進行趨勢變化預測。

#### 資料準備
使用**台灣電力公司_本年度每日尖峰備轉容量率**資料欄位中的「日期」作為模型所需的「ds」欄位，以「備轉容量(萬瓩)」作為模型所需的「y」欄位，接著再針對這個「y」欄位進行正規化，再丟進模型中進行訓練。

#### 參數設定
在`holidays`參數中放入2021年1月1日 ~ 2021年3月21日，台灣周末假期與國定假日的資料。

#### 訓練結果
我們從[台電未來一週電力供需預測](https://www.taipower.com.tw/tc/page.aspx?mid=209)中蒐集2021年3月23日 ~ 2021年3月29日的預測資料作為我們的驗證集，最終模型預測出來的結果經RMSE的計算後可得到96.048，圖表則如下所示：
![](https://i.imgur.com/hze1OHy.png)