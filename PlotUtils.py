import matplotlib.pyplot as plt
import numpy as np

# 畫折線圖
def hydrograph(obv, est):
    fig, ax = plt.subplots()  
    ax.plot(obv, color='red', label='Observation')
    ax.plot(est, color='blue', label='Estimation')
    ax.set_title('Hydrograph')
    ax.set_xlabel('Time(hr)')
    ax.set_ylabel('Depth(cm)')
    ax.legend()
    ax.grid()
    return fig
  
# 畫45度線圖
def scatter(obv, est):
    fig, ax = plt.subplots()  
    minVal = min(min(obv), min(est))
    maxVal = max(max(obv), max(est))
    line_range = [minVal, maxVal]
    ax.plot(line_range, line_range, color="red", linewidth=1, linestyle='-')
    # ax.set_xlim(tick_range)
    # ax.set_ylim(tick_range)
    ax.scatter(obv, est, c=np.random.rand(len(obv)), cmap='rainbow', s=20)
    ax.set_title('Scatter Plot')
    ax.set_xlabel('Observation')
    ax.set_ylabel('Estimation')
    ax.grid()
    return fig

# 畫出所有所需圖片並存檔
def draw_all(fig_names, fig_folders, event_num, Y_test, Y_predict, save=True):
    for fig_name in fig_names:
        fig_path = fig_folders[fig_name] + f'RES-test_EV' + "%02d" % (event_num+1) + '.png'
        if fig_name == 'Hydrograph':
            fig = hydrograph(Y_test, Y_predict)
        elif fig_name == 'Scatter plot':
            fig = scatter(Y_test, Y_predict)
        if save:
            fig.savefig(fig_path)
            print(f'{fig_name} saved in ' + fig_path)
