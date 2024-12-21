import os

# 输入目录
input_txt_dir = r'F:\Download\机器-订单转写结果_202409297622695754'
output_audio_dir = r'F:\Download\GPT-SoVITS-beta\GPT-SoVITS-beta0706\output\slicer_opt'

# 输出文件路径
output_list_file = r'生成的list文件.list'

# 初始化一个空列表，用于存储所有生成的条目
output_lines = []

# 遍历输入目录下的所有txt文件
for txt_filename in os.listdir(input_txt_dir):
    if txt_filename.endswith('.txt'):
        txt_file_path = os.path.join(input_txt_dir, txt_filename)

        with open(txt_file_path, 'r', encoding='utf-8-sig') as file:  # 使用 utf-8-sig 来移除 BOM
            lines = file.readlines()

            # 解析txt文件的内容（第一行是音频文件名，第二行是字幕）
            for i in range(0, len(lines), 2):
                audio_filename = lines[i].strip()  # 第一行是音频文件名
                subtitle = lines[i + 1].strip()  # 第二行是字幕标注

                # 构建最终格式的字符串
                audio_file_path = os.path.join(output_audio_dir, audio_filename)

                # 检查文件是否存在
                if os.path.exists(audio_file_path):
                    # 替换为日语标识符 JA
                    output_line = f"{audio_file_path}|slicer_opt|JA|{subtitle}"
                    output_lines.append(output_line)
                else:
                    print(f"文件未找到: {audio_file_path}")  # 打印未找到的文件

# 将结果写入输出文件
with open(output_list_file, 'w', encoding='utf-8') as output_file:
    for line in output_lines:
        output_file.write(line + '\n')

print("List 文件生成完毕，保存至:", output_list_file)
