from flask import render_template
from app import app

# 导入其他路由模块
from app.routes.save_video import save_video_function
from app.routes.get_video import get_video_function
from app.routes.get_result import get_result_function
from app.routes.change import change_function
# push_pose_to_GCN，将客户端的pose数据传递给GCN，提取特征，整个程序的运行device设置也写在这里
from app.routes.push_pose_to_GCN import push_pose_to_GCN_function 
from app.routes.push_sample_video_pose import push_sample_video_pose_to_GCN_and_save_function
from app.routes.video_pose_segmentation_by_sample_feature import video_pose_segmentation_by_sample_feature_function
from app.routes.getSampleVideoInformation import getSampleVideoInformation_function

# 针对不同 URL 注册路由
app.add_url_rule('/', view_func=lambda: render_template('index.html'))
app.add_url_rule('/save_video', view_func=save_video_function)
app.add_url_rule('/get_video', view_func=get_video_function)
app.add_url_rule('/get_result', view_func=get_result_function)
app.add_url_rule('/change', view_func=change_function)
# push_pose_to_GCN，将客户端待检测视频对应的pose数据传递给GCN，并通过与标准样例对比进行切分
app.add_url_rule('/video_pose_segmentation_by_sample_feature', view_func=video_pose_segmentation_by_sample_feature_function, methods=['POST']) 
# push_sample_video_pose，接收客户端传来的标准视频样例对应的pose数据
app.add_url_rule('/push_sample_video_pose_to_GCN_and_save', view_func=push_sample_video_pose_to_GCN_and_save_function, methods=['POST']) 
# getSampleVideoInformation 获取样例信息
app.add_url_rule('/getSampleVideoInformation', view_func=getSampleVideoInformation_function, methods=['get']) 