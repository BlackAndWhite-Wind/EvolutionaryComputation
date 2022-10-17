import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

max_epochs = 400  # 最大的迭代次数
_cmp = 0.95  # 种群交叉的概率
_mop = 0.05  # 种群变异的概率

fun_one_bound = [-5.12, 5.12]
fun_two_bound = [-2.048, 2.048]
fun_four_bound = [-65.536, 65.536]
fun_six_bound = [-5.12, 5.12]


best_fitness = []  # 每一代的最好的适应度
all_fitness = []  # 所有代所有个体的适应度
one_fitness = []  # 某一代所有个体的适应度

# 初始化最开始的种群规模
first_population = np.random.randint(low=0, high=2, size=(200, 2, 20))


def function_one(x1, x2):  # 范围：-5.12<= x <= 5.12
    return x1 ** 2 + x2 ** 2


def function_two(x1, x2):  # 范围：-2.048<= x <= 2.048
    return 100 * (x1 ** 2 - x2) ** 2 + (1 - x1) ** 2


def function_four(x):  # 范围：-65.536<= x <= 65.536
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                   [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = np.zeros(25)
    for j in range(0, 25):
        bS[j] = np.sum((x.T-aS[:, j])**6)

    return (1/500+np.sum(1/(np.arange(1, 25+1)+bS)))**(-1)


def function_six(x1,x2):  # 范围：-5.12<= x <= 5.12
    return 20+x1*x1+x2*x2-10*np.cos(2*np.pi*x1)-10*np.cos(2*np.pi*x2)


# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.arange(-2.048, 2.048, 0.1)
# Y = np.arange(-2.048, 2.048, 0.1)
# X, Y = np.meshgrid(X, Y)
# Z = 100 * (X ** 2 - Y) ** 2 + (1 - X) ** 2

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
# ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
# ax.set_zlim(0, 2000)
# plt.show()


def reflect_fuc(each, func_bound):
    # 传入每一对以及函数的定义域范围。
    # 建立二进制编码到一个实数的映射。
    x_, y_ = 0, 0

    for i in range(10):
        x_ += each[0][i]*(2**i)
    for i in range(10):
        y_ += each[1][i]*(2**i)

    fin_x = (x_/(2**10-1))*(func_bound[1]-func_bound[0])+func_bound[0]
    fin_y = (y_/(2**10-1))*(func_bound[1]-func_bound[0])+func_bound[0]
    return fin_x, fin_y

def get_fitness(first_population, function, func_bound):
    # 获取所有个体的适应度
    # 传入初始的种群，函数以及函数的定义域
    each_fitness = []
    for each in first_population:
        x_1, x_2 = reflect_fuc(each, func_bound)  # 映射得到x1,x2
        each_fitness.append(function(x_1, x_2))  # 添加到fitness
    return each_fitness


def rws_algoritm(first_population, fitness, n):  # 定义轮盘赌算法
    next_population = []  # 定义下一个子代
    sum_ = sum(fitness)  # 获取所有的适应度的和

    p_ = ((sum_-fitness)/sum_) / (len(fitness)-1)  # 获得概率

    idx = np.random.choice(
        np.arange(len(first_population)), size=n, replace=True, p=p_)

    for i in idx:
        next_population.append(first_population[i])
    return next_population


def mutation(first_population, mop):  # 变异
    next_population = []  # 定义下一代
    for each in first_population:  # 遍历
        res = each  # 子代变成父代(进行更新)
        if np.random.rand() < mop:  # 随机初始化一个数 比他大不发生变化 比他小就发生交叉交换
            # 定义x_mu_place 是发生变异的位置
            x_mu_place = np.random.randint(0, len(each[0]))
            res[0][x_mu_place] = abs(res[0][x_mu_place]-1)

        if np.random.rand() < mop:  # 随机初始化一个数 比他大不发生变化 比他小就发生交叉交换
            # 同理，y_place 也是一样的。
            y_mu_place = np.random.randint(0, len(each[1]))
            res[1][y_mu_place] = abs(res[1][y_mu_place]-1)

        next_population.append(res)
    return next_population


def cross_mutation(first_population, cmp):  # 交叉与变异
    next_population = []  # 定义下一代
    for each in first_population:  # 遍历
        res = each  # 子代变成父代
        if np.random.rand() < cmp:  # 随机初始化一个数 比他大不发生变化 比他小就发生交叉交换
            # 随机从first_population中生成一段，然后进行交换
            random_length = first_population[np.random.randint(
                0, len(first_population))]  # 另一个

            # 定义随机生成的x交叉点和y交叉点
            x_c_m_pos = np.random.randint(0, len(random_length[0]))
            y_c_m_pos = np.random.randint(0, len(random_length[1]))

            for i in range(x_c_m_pos, len(random_length[0])):
                res[0][i] = random_length[0][i]  # 赋值,剪切之后交换
            for j in range(y_c_m_pos, len(random_length[1])):
                res[1][j] = random_length[1][j]  # 赋值,剪切之后交换

        next_population.append(res)
    return next_population


def nature_selection(function, parent, cop, mop, fun_bound):  # 自然选择

    next_population, fitness, best_best = [], [], []  # 下一代 && 适应度 && 精英

    fitness = get_fitness(parent, function, fun_bound)  # 获取适应度
    best_idx = fitness.index(np.min(fitness))  # 找到最小的那个
    best_best = parent[best_idx].copy()  # 找到最好的那个值，下标，当作是精英

    # 接下来进行变异，交叉和变异，轮盘选择，再进行fitness的测试
    # 每一个产生的next_作为下一个输入的种群

    next_population = mutation(parent, mop)
    next_population = cross_mutation(next_population, cop)
    next_population = rws_algoritm(
        next_population, fitness, len(next_population)-1)

    next_population.append(best_best)  # 添加精英进去

    fitness = get_fitness(next_population, function, fun_bound)  # 再一次获取适应度

    return next_population


for i in range(max_epochs):   # 最大的迭代次数
    first_population = nature_selection(
        function_six, first_population, _cmp, _mop, fun_six_bound)
    one_fitness = get_fitness(first_population, function_six, fun_six_bound)
    all_fitness.append(one_fitness)
    best_fitness.append(np.min(one_fitness))

print("finish")

# 画图
print("average:{}".format(sum(best_fitness)/len(best_fitness)))
print("max_element:{}".format(max(best_fitness)))
print("min_element:{}".format(min(best_fitness)))

plt.plot(np.arange(max_epochs), best_fitness)
plt.xlabel("epoches")
plt.ylabel("best_fitness")
plt.show()

# 画出每一代的图形
plt.boxplot(all_fitness, labels=np.arange(max_epochs),
            sym="+",
            widths=0.6,    # 指定箱线图的宽度，默认为0.5；
            patch_artist=True,    # 是否填充箱体的颜色；
            showmeans=True
            )
plt.show()
