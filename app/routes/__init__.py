from flask import render_template
from app import app

# 导入其他路由模块
from app.routes.save_video import save_video_function
from app.routes.get_video import get_video_function
from app.routes.get_result import get_result_function
from app.routes.change import change_function
from app.routes.try_get_data import try_get_data_function

# 针对不同 URL 注册路由
app.add_url_rule('/', view_func=lambda: render_template('index.html'))
app.add_url_rule('/save_video', view_func=save_video_function)
app.add_url_rule('/get_video', view_func=get_video_function)
app.add_url_rule('/get_result', view_func=get_result_function)
app.add_url_rule('/change', view_func=change_function)
app.add_url_rule('/try_get_data', view_func=try_get_data_function, methods=['POST'])
