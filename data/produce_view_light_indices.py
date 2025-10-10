import os


def load_light_index(file_path):
    """ Reads the light index file and returns a dictionary {num_of_light: [light_idx1, light_idx2, ...]} """
    light_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            values = list(map(int, line.strip().split()))  # 转换为整数列表
            if len(values) > 1:
                num_of_light = values[0]  # 第一个数字是 num_of_light
                light_dict[num_of_light] = values[1:]  # 剩下的是 light index, 从 0 开始
    return light_dict

light_dict = load_light_index("../configs/light_indices.txt")
view_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                12, 13, 14, 15, 16, 17, 18, 19]
test_view_indices = [1, 11]
num_views = len(view_indices)

for num_of_light, light_indices in light_dict.items():
    print("num_of_light: ", num_of_light)
    print("light_indices: ", light_indices)
    print()
    test_light_indices = list(set(range(1, 97)) - set(light_indices))

    txt_path = f"view_light_indices/geo_eval_view_{num_views}_light_{num_of_light}_train_test.txt"
    if os.path.exists(txt_path):
        os.remove(txt_path)

    with open(txt_path, "w") as f:
        for view_idx in view_indices:
            for light_idx in light_indices:
                f.write(f"V{view_idx:02d}L{light_idx-1:02d}\n")


    with open(txt_path, "a") as f:
        for view_idx in test_view_indices:
            for light_idx in test_light_indices:
                f.write(f"V{view_idx:02d}L{light_idx - 1:02d}\n")

