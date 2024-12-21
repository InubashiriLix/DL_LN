# 纯手工实现的 powerset 函数
def manual_powerset(s):
    s = list(s)  # 将集合转换为列表
    powerset = [[]]  # 初始化 powerset，包含空集

    for elem in s:
        # 将当前的 powerset 与每个新元素生成新的子集
        new_subsets = [subset + [elem] for subset in powerset]
        powerset.extend(new_subsets)

    return powerset


# 示例：寻找集合 {1, 2, 3} 的所有子集
example_set = {1, 2, 3, 4}
manual_powerset_result = manual_powerset(example_set)
print(manual_powerset_result)
