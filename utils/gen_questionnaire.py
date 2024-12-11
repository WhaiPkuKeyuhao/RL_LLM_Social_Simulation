import pandas as pd

data = []
with open("data/prompt_test.txt", "r") as f:
    for line in f:
        data.append(line)

formatted_data = []
for idx, d in enumerate(data):
    row = {"id": idx}
    row["question"] = d
    row["score"] = None
    row["reason"] = None

    formatted_data.append(row)

bus_routes_neighbors_df = pd.DataFrame(formatted_data)
bus_routes_neighbors_df.to_csv("data/人类反馈智能体活动评分问卷.csv", index=False, encoding='utf-8-sig')
