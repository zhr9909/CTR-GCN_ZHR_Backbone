from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)  # 在调试模式下运行 Flask 应用
