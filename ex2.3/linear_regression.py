import numpy as np

# y = wx + b
def compute_error_for_line_given_points(b, w, points):
    # 根据当前的w,b 参数计算均方差损失
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]# 获得i号点的输入x
        y = points[i, 1]# 获得j号点的输入x
        # 计算差的平方，并累加
        totalError += (y - (w * x + b)) ** 2
        # 将累加的误差求平均，得到均方差
    return totalError / float(len(points))

def step_gradient(b_current, w_current, points, learningRate):
    # 计算误差函数在所有点上的导数，并更新w,b
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))# 总样本数
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 误差函数对b 的导数：grad_b = 2(wx+b-y)，
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        # 误差函数对w 的导数：grad_w = 2(wx+b-y)*x，
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    # 根据梯度下降算法更新 w',b',其中lr 为学习率
    new_b = b_current - (learningRate * b_gradient)
    new_m = w_current - (learningRate * w_gradient)
    return [new_b, new_m]

#梯度更新
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # 循环更新w,b多次
    b = starting_b
    m = starting_m
    # 根据梯度下降算法更新多次
    for i in range(num_iterations):
        # 计算梯度并更新一次
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        loss = compute_error_for_line_given_points(b, m, points)
        if i%50 == 0: # 打印误差和实时的w,b 值
            print(f"iteration:{i}, loss:{loss}, w:{m}, b:{b}")
    return [b, m]

def run():
    points = np.genfromtxt("./data/data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m,
                  compute_error_for_line_given_points(initial_b, initial_m, points))
          )
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )

if __name__ == '__main__':
    run()