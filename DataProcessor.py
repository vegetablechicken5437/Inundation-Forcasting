import numpy as np
import pandas as pd
import openpyxl
import AD2SD as AVT

# 載入資料
def load_data(AD_FILE, IOLAG_FILE):
    ad = AVT.read_AD(AD_FILE)
    iolag = AVT.read_IOLag(IOLAG_FILE)
    sd = AVT.ad2sd(ad, iolag)   
    events = [pd.DataFrame(sd[i]) for i in range(len(sd))]
    return events

def load_SD(SD_FILE):
    wb = openpyxl.load_workbook(SD_FILE, data_only=True)
    events = [wb[sheetname] for sheetname in wb.sheetnames]
    for i in range(len(events)):
        events[i] = pd.DataFrame(get_values(events[i]))
    return events

# 取得資料值並儲存為列表
def get_values(sheet):
    arr = [] 
    for row in sheet:
        temp = []  
        for column in row:
            temp.append(column.value)
        arr.append(temp)
    return arr

# 正規化資料(將資料範圍限縮在0~1之間，防止資料大小差異過大，影響模型訓練)
def normalize(data):
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data

# 反正規化資料(將正規化的資料還原，用於結果呈現)
def denormalize(data, original):
    denormalized_data = data * (np.max(original) - np.min(original)) + np.min(original)
    return denormalized_data

# 分割訓練及測試資料
def split_data(X, Y, split_boundary):
    X_train = X[:split_boundary]
    Y_train = Y[:split_boundary]
    X_test = X[split_boundary:]
    Y_test = Y[split_boundary:]
    return X_train, Y_train, X_test, Y_test

# 產生可輸入模型的資料及標籤
def create_sequences(events):
    df = pd.concat(events, ignore_index=True)
    ref = df.iloc[:, :(len(df.columns)-1)] # 取得第一欄~倒數第二欄的資料為參考資料
    pred = df.iloc[:, (len(df.columns)-1)] # 以最後一欄的資料為目標資料

    X, Y = [], []
    num_data = len(ref)
    for i in range(num_data):
        seq = ref.iloc[i, :]
        label = pred[i]
        X.append(seq)
        Y.append(label)

    X, Y = np.array(X), np.array(Y)
    X = X.reshape(X.shape[0], X.shape[1], 1).astype('float32')
    Y = Y.reshape(Y.shape[0], 1).astype('float32')

    return X, Y

# 整理輸入資料順序以便輸入模型
def reorder_events(events, event_order):
    test_event = events[0] # 取得第一場事件資料為測試資料
    events = events[1:] # 第二場~最後一場
    events.append(test_event) # 將測試資料移到陣列尾端

    temp = event_order[0]
    event_order = event_order[1:]
    event_order.append(temp)

    boundary = []
    num_events = len(events)
    for i in range(num_events):
        boundary.append(sum([len(events[j]) for j in range(i+1)]))
    split_boundary = boundary[-2] # 訓練與測試資料的分割邊界

    return events, event_order, boundary, split_boundary

def neg2zero(data):
    return np.array([max(0, num) for num in data], dtype=np.float32)

def get_evMaxVal(Y, boundaries):
    indexOfMaxVal = list(Y).index(np.max(Y)) # 計算 event with max value
    for i in range(len(boundaries[0])):
        if indexOfMaxVal < boundaries[0][i]:
            evMaxVal = i+1
            break
    return evMaxVal