#!/bin/bash

# 检查输入参数
if [ $# -eq 0 ]; then
    echo "用法: $0 start|stop"
    exit 1
fi

# 定义文件名
NOTEBOOK="train.ipynb"
SCRIPT="train.py"

# 启动训练
start() {
    # 转换 Jupyter Notebook 为 Python 脚本
    jupyter nbconvert --to script "$NOTEBOOK"

    # 使用 nohup 在后台运行 Python 脚本，输出到 nohup.out
    nohup python "$SCRIPT" &

    # 输出进程 ID
    echo "训练已启动，进程 ID: $!"
}

# 停止训练
stop() {
    # 查找 Python 进程并终止
    PID=$(ps aux | grep "$SCRIPT" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PID" ]; then
        echo "没有找到正在运行的训练进程。"
    else
        kill -9 "$PID"
        echo "已终止进程 ID: $PID"
    fi
}

# 根据参数选择启动或停止
case $1 in
    start)
        start
        ;;
    stop)
        stop
        ;;
    *)
        echo "用法: $0 start|stop"
        exit 1
        ;;
esac
