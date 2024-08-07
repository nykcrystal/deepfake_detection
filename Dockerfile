# 使用官方的Python基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 安装 uWSGI
RUN pip install uwsgi

# 创建必要的目录并设置权限
RUN mkdir -p /app/deepfake_detection/logs && chown -R www-data:www-data /app/deepfake_detection

# 复制项目文件到工作目录
COPY . /app

# 暴露端口（假设你的应用在9090端口运行）
EXPOSE 5004

# 启动应用（使用uWSGI）
CMD ["uwsgi", "--ini", "uwsgi.ini"]
