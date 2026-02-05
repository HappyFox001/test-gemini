#!/bin/bash

# SSH 登录脚本
# 服务器: 18.182.56.215

echo "正在使用 ResonAI.pem 登录到 18.182.56.215..."
echo ""
echo "如果不知道用户名，请尝试以下命令："
echo ""
echo "1. Ubuntu 系统:"
echo "   ssh -i ResonAI.pem ubuntu@18.182.56.215"
echo ""
echo "2. Amazon Linux:"
echo "   ssh -i ResonAI.pem ec2-user@18.182.56.215"
echo ""
echo "3. Debian 系统:"
echo "   ssh -i ResonAI.pem admin@18.182.56.215"
echo ""
echo "4. CentOS 系统:"
echo "   ssh -i ResonAI.pem centos@18.182.56.215"
echo ""
echo "5. Root 用户:"
echo "   ssh -i ResonAI.pem root@18.182.56.215"
echo ""
echo "----------------------------------------"
echo "使用正确的用户名: ec2-user"
echo ""

# 使用 ec2-user 用户名（Amazon Linux）
ssh -i ResonAI.pem ec2-user@18.182.56.215
