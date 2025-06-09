from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = '12ecc519-f814-4ce9-a5d4-f61e12811029'  # 从步骤2获取
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)  # 登录

# 替换为你的模型ID（格式：你的用户名/模型名）
model_id = "PkuWhAIKyh/HumanActionGrader"
model_dir = "fine_tuned_model"  # 本地模型文件夹路径

api.upload_folder(
    repo_id=model_id,
    folder_path=model_dir,
    commit_message='upload model folder to repo',
)