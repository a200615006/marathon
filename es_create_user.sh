#!/bin/bash
# Elasticsearch 专用用户创建 + 文件夹迁移脚本（无需复杂权限配置）

# 可修改参数
ES_USER="es_user"                  # ES专用用户名
ES_SOURCE_DIR="./elasticsearch-9.1.3"  # 源ES解压目录
ES_TARGET_DIR="/home/$ES_USER/elasticsearch-9.1.3"  # 目标路径（用户家目录下）

# 1. 检查ES源目录是否存在
if [ ! -d "$ES_SOURCE_DIR" ]; then
    echo "错误：ES源目录 $ES_SOURCE_DIR 不存在，请先解压：tar -xvf elasticsearch-9.1.3.tar.gz"
    exit 1
fi

# 2. 判断权限执行前缀（适配sudo/root环境）
if command -v sudo &> /dev/null; then
    SUDO="sudo"
else
    if [ "$(id -u)" -ne 0 ]; then
        echo "错误：无sudo且非root用户，无法创建用户/移动文件夹"
        exit 1
    fi
    SUDO=""
fi

# 3. 检查并创建用户（确保家目录存在）
if id -u "$ES_USER" &> /dev/null; then
    echo "✅ 用户 $ES_USER 已存在，跳过创建"
else
    echo "🔧 正在创建ES专用用户 $ES_USER..."
    $SUDO useradd -m $ES_USER  # -m强制创建家目录
    if [ $? -ne 0 ]; then
        echo "❌ 用户创建失败"
        exit 1
    fi
    # 设置用户密码
    echo "请为 $ES_USER 设置密码（后续切换用户需使用）："
    $SUDO passwd $ES_USER || { echo "❌ 密码设置失败"; exit 1; }
    echo "✅ 用户 $ES_USER 创建成功"
fi

# 4. 移动ES文件夹到用户家目录（核心修改：替换权限配置为剪切操作）
if [ -d "$ES_TARGET_DIR" ]; then
    echo "⚠️ 目标目录 $ES_TARGET_DIR 已存在，跳过移动"
else
    echo "🔧 正在将ES目录迁移到 $ES_USER 家目录..."
    $SUDO mv "$ES_SOURCE_DIR" "$ES_TARGET_DIR" || { echo "❌ 文件夹迁移失败"; exit 1; }
    # 确保用户对迁移后的目录有完全所有权（移动后可能继承原权限，保险起见修复）
    $SUDO chown -R $ES_USER:$ES_USER "$ES_TARGET_DIR"
    echo "✅ ES目录已迁移至 $ES_TARGET_DIR"
fi

# 5. 输出修正后的操作指引（路径已更新为用户家目录）
echo -e "\n📝 下一步操作："
echo "1. 打开新终端窗口"
echo "2. 切换到ES用户：su - $ES_USER（输入刚才设置的密码）"
echo "3. 进入ES启动目录：cd elasticsearch-9.1.3/bin（因已在用户家目录，无需绝对路径）"
echo "4. 启动ES服务：./elasticsearch（记录终端显示的默认密码）"
