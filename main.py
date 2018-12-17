import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from IPython.display import HTML


%matplotlib inline

fig = plt.figure(figsize=(5,5))
ax  = plt.gca()

############# 迷路描画 #############
plt.plot([1, 1], [0, 1], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)
plt.plot([2, 2], [2, 1], color='red', linewidth=2)
plt.plot([2, 3], [1, 1], color='red', linewidth=2)

# 目盛り設定
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')

plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')

plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')

plt.text(0.5, 2.3, 'start', size=14, ha='center')


# 現在地の描画
line, = ax.plot([0.5, 0.5], [2.5, 2.5], marker="o", color="g", markersize="60")
############# 迷路描画 #############



theta_0 = np.array([[np.nan, 1, 1, np.nan], # s0
                    [np.nan, 1, np.nan, 1], # s1
                    [np.nan, np.nan, 1, 1], # s2
                    [1, 1, 1, np.nan], # s3
                    [np.nan, np.nan, 1, 1], # s4
                    [1, np.nan, np.nan, np.nan], # s5
                    [1, np.nan, np.nan, np.nan], # s6
                    [1, 1, np.nan, np.nan], # s7
                    ])


# 行動方策piを変換する

def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape # 行例のサイズを取得
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :]) # 割合の計算

    pi = np.nan_to_num(pi) # nanを0に変換

    return pi

# 初期の方策pi_0を求める
# 壁方向の移動確率は0
# その他の方向へは同じ確率で移動するように変換
pi_0 = simple_convert_into_pi_from_theta(theta_0)

print(pi_0)

# 現在地の数値化
def get_next_s(pi, s):
    direction = ["up", "right", "down", "left"]

    # pi[s, :]の確率に従って移動
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        s_next = s - 3
    elif next_direction == "right":
        s_next = s + 1
    elif next_direction == "down":
        s_next = s + 3
    elif next_direction == "left":
        s_next = s - 1

    return s_next

# ゴールするまでの移動の関数
def goal_maze(pi):
    s = 0
    state_history = [0] # エージェントの移動を

    while (True):
        next_s = get_next_s(pi, s)
        state_history.append(next_s)

        if next_s == 8:
            break
        else:
            s = next_s
    return state_history

state_history = goal_maze(pi_0)
print(state_history)
print(f"step: {len(state_history)}")

# エージェントの移動を可視化

def init():
    line.set_data([], [])
    return (line, )

def animate(i):
    state = state_history[i]
    x = (state % 3) + 0.5
    y = 2.5 - int(state / 3)
    line.set_data(x, y)
    return (line,)

# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history), interval=200, repeat=False)
# HTML(anim.to_jshtml())


####### 方策反復方  #######

def softmax_convert_into_pi_from_theta(theta):
    beta = 1.0
    [m, n] = theta.shape
    pi = np.zeros((m, n))

    exp_theta = np.exp(beta * theta)
    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])

    pi = np.nan_to_num(pi) # nanを0に変換
    return pi

pi_0 = softmax_convert_into_pi_from_theta(theta_0)
print(pi_0)


def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        action = 0
        s_next = s -3
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "down":
        action = 2
        s_next = s + 3
    elif next_direction == "left":
        action = 3
        s_next = s - 1

    return [action, s_next]

# ゴールまでループ
def goal_maze_ret_s_a(pi):
    s = 0
    s_a_history = [[0, np.nan]]

    while(True):
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action

        s_a_history.append([next_s, np.nan])

        if next_s == 8:
            break
        else:
            s = next_s

    return s_a_history

s_a_history = goal_maze_ret_s_a(pi_0)
print(s_a_history)
print(f"step: {len(s_a_history)}")


# thetaの更新関数
def update_theta(theta, pi, s_a_history):
    eta = 0.1 # 学習率
    T = len(s_a_history) - 1

    [m, n] = theta.shape
    delta_theta = theta.copy()

    # delta_thetaを要素ごとに求める
    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i, j])):
                SA_i = [SA for SA in s_a_history if SA[0] == i]
                SA_ij = [SA for SA in s_a_history if SA == [i, j]]

                N_i = len(SA_i)
                N_ij = len(SA_ij)
                delta_theta[i, j] = (N_ij + pi[i, j] * N_i) / T
    new_theta = theta + eta * delta_theta

    return new_theta

new_theta = update_theta(theta_0, pi_0, s_a_history)
pi = softmax_convert_into_pi_from_theta(new_theta)
print(pi)

stop_epsilon = 10 ** -8 # 10^-8よりも方策に変化が少なかったら学習終了
theta = theta_0
pi = pi_0

is_continue = True
count = 1
while  is_continue:
    s_a_history = goal_maze_ret_s_a(pi)
    new_theta = update_theta(theta, pi, s_a_history)
    new_pi = softmax_convert_into_pi_from_theta(new_theta)

    print(np.sum(np.abs(new_pi - pi))) # 方策の変化量
    print(f"step: {str(len(s_a_history) - 1)}")

    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        is_continue = False
    else:
        theta = new_theta
        pi = new_pi

np.set_printoptions(precision=3, suppress=True)
print(pi)

####### 方策反復方  #######
