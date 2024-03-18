import DataProcessor as dp
from MyModels import My_LSTM
import PlotUtils as pltUT
import RES_Gen
import IndexProcessor as idp
import os

# 檔案、資料夾路徑參數
AD_FILE = 'data/AD.xlsx'
IOLAG_FILE = 'data/IOLag.xlsx'   
SD_FILE = 'data/SD.xlsx'                                   # 資料檔案路徑
WEIGHTS_FOLDER = 'weights/'                                # 權重檔的資料夾路徑
OUTPUT_FOLDER = 'outputs/'                                 # 輸出結果的資料夾路徑
HYDROGRAPH_FOLDER = 'outputs/Hydrographs/'                 # 歷線圖的資料夾路徑
SCATTER_PLOT_FOLDER = 'outputs/Scatter_Plots/'             # 散點圖的資料夾路徑
RES_train_path = OUTPUT_FOLDER + 'RES-train.xlsx'
RES_test_path = OUTPUT_FOLDER + 'RES-test.xlsx'
Index_path = OUTPUT_FOLDER + 'Index.xlsx'

if not os.path.exists(HYDROGRAPH_FOLDER):
    os.makedirs(HYDROGRAPH_FOLDER)
if not os.path.exists(SCATTER_PLOT_FOLDER):
    os.makedirs(SCATTER_PLOT_FOLDER)

# 模型參數
epochs = 10             # 迭代次數(要讓模型看幾次訓練資料)
batch_size = 16         # 批次大小(每看多少筆資料更新一次權重)
lr = 0.001              # 學習速率
loss_fn = 'mse'         # 損失函數

events = dp.load_data(AD_FILE, IOLAG_FILE)              # 載入資料
num_events = len(events)                                # 事件數量
event_order = list(range(1, num_events+1))              # 事件排列順序

boundaries = []     # 用於儲存事件分割邊界
event_orders = []   # 用於儲存每次訓練的事件排列順序

RES_test = []       # 用於儲存測試資料的預測結果

# 交叉驗證(使不同事件輪流當測試資料)
for i in range(num_events):
    events, event_order, boundary, split_boundary = dp.reorder_events(events, event_order)  # 整理輸入資料順序以便輸入模型
    event_orders.append(event_order)                                                        # 紀錄資料的順序  
    boundaries.append(boundary)                                                             # 紀錄資料長度的分割邊界 

    # 產生輸入模型的資料
    X, Y = dp.create_sequences(events)                                          # 產生參考資料序列(X)及標籤資料序列(Y)
    nX, nY = dp.normalize(X), dp.normalize(Y)                                   # 正規化
    X_train, Y_train, X_test, Y_test = dp.split_data(nX, nY, split_boundary)    # 分割訓練及測試資料
    weights_path = WEIGHTS_FOLDER + 'Weights_EV' + "%02d" % (i+1) + '.h5'       # 輸出的權重檔路徑

    print('\n[第 %d/%d 次訓練]' %(i+1, num_events))
    print('▶ 以第 ' + (', '.join(str(x) for x in sorted(event_order[:-1]))) + ' 場事件為訓練資料，以第 ' + str(event_order[-1]) + ' 場事件為測試資料\n')

    # 創建模型
    model = My_LSTM(X_train.shape[1], X_train.shape[2])
    # 訓練模型
    model = model.train(X_train, Y_train, X_test, Y_test, lr, loss_fn, epochs, batch_size, show_training_history=False)
    # 儲存模型
    model.save(weights_path)
    print('Training weights saved in ' + weights_path)

    # 預測
    Y_predict = model.predict(X_test)
    # 反正規化並將所有負值改為0
    obv_test = dp.neg2zero(dp.denormalize(Y_test, Y))
    est_test = dp.neg2zero(dp.denormalize(Y_predict, Y))
    # 儲存預測結果
    RES_test.append([obv_test, est_test]) 

    #***************************************************************************
    # 畫圖
    #***************************************************************************
    fig_folders = {'Hydrograph':HYDROGRAPH_FOLDER, 'Scatter plot':SCATTER_PLOT_FOLDER}
    pltUT.draw_all(list(fig_folders.keys()), fig_folders, i, obv_test, est_test)


evMaxVal = dp.get_evMaxVal(Y, boundaries)

RES_Gen.gen_RES_test(RES_test, events, evMaxVal, RES_test_path)

index_names = ['RMSE', 'MAE', 'CE', 'CC', 'EQp', 'ETp']   # 要計算的指標

index_test = idp.get_all_indices(num_events, RES_test, index_names)
idp.write_Index(num_events, evMaxVal, index_names, index_test, Index_path)