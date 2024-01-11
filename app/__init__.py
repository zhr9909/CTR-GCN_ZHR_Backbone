from flask import Flask

app = Flask(__name__)

# 导入路由模块
from app.routes import save_video, get_video, get_result, change
