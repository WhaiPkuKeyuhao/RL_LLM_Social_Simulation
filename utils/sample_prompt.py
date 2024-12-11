import random
### 根据rewards.py print(_content) 打印 随机采样100条提示词
data = []
with open("log.txt", "r") as f:
    for line in f:
        if "这是" in line:
            data.append(line)

print(len(data))
random.shuffle(data)

with open("data/prompt_test.txt", "w") as f:
    for d in data[:100]:
        f.write(d)

