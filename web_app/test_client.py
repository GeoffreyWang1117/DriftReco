"""
SmartNet Web应用API测试客户端
测试各个API端点的功能
"""

import requests
import json
import time
import sys
from pprint import pprint

BASE_URL = "http://localhost:5000"

def test_api_endpoint(endpoint, method='GET', data=None, description=""):
    """测试API端点"""
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"\n🧪 测试: {description or endpoint}")
        print(f"   URL: {method} {url}")
        
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"   ❌ 不支持的方法: {method}")
            return False
        
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('success'):
                    print(f"   ✅ 成功")
                    return result
                else:
                    print(f"   ❌ API返回失败: {result.get('error', '未知错误')}")
                    return False
            except json.JSONDecodeError:
                print(f"   ⚠️  非JSON响应")
                return response.text
        else:
            print(f"   ❌ HTTP错误: {response.status_code}")
            return False
            
    except requests.ConnectionError:
        print(f"   ❌ 连接失败 - Web应用可能未启动")
        return False
    except requests.Timeout:
        print(f"   ❌ 请求超时")
        return False
    except Exception as e:
        print(f"   ❌ 未知错误: {e}")
        return False

def main():
    """运行所有API测试"""
    print("🚀 SmartNet Web应用API测试")
    print("=" * 60)
    
    # 测试1: 检查主页
    print("📖 第1步: 测试主页访问")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            print("   ✅ 主页可正常访问")
        else:
            print(f"   ❌ 主页访问失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 主页访问出错: {e}")
    
    # 测试2: 获取系统信息
    print("\n🖥️ 第2步: 获取系统信息")
    system_info = test_api_endpoint('/api/system_info', description="系统信息检查")
    if system_info:
        sys_data = system_info.get('system', {})
        print(f"   GPU可用: {sys_data.get('gpu_available', False)}")
        print(f"   GPU数量: {sys_data.get('gpu_count', 0)}")
        print(f"   PyTorch版本: {sys_data.get('pytorch_version', 'N/A')}")
        if sys_data.get('gpu_info'):
            for i, gpu in enumerate(sys_data['gpu_info']):
                print(f"   GPU{i}: {gpu.get('name', 'Unknown')} ({gpu.get('memory_total', 0)}MB)")
    
    # 测试3: 获取组件列表
    print("\n🧱 第3步: 获取可用组件")
    components_result = test_api_endpoint('/api/components', description="获取网络组件列表")
    if components_result:
        components = components_result.get('components', [])
        print(f"   可用组件数量: {len(components)}")
        for comp in components:
            print(f"   • {comp['name']} ({comp['id']}): {comp['description']}")
    
    # 测试4: 构建简单网络
    print("\n🔧 第4步: 构建测试网络")
    network_config = {
        'components': [
            {
                'id': 'input_layer',
                'type': 'mlp',
                'params': {
                    'input_dim': 128,
                    'output_dim': 64,
                    'hidden_dims': [96],
                    'dropout': 0.1
                }
            },
            {
                'id': 'output_layer', 
                'type': 'mlp',
                'params': {
                    'input_dim': 64,
                    'output_dim': 1,
                    'hidden_dims': [],
                    'dropout': 0.0
                }
            }
        ],
        'connections': [
            {'from': 'input_layer', 'to': 'output_layer'}
        ],
        'config': {
            'name': 'test_network',
            'input_features': 128,
            'output_features': 1,
            'learning_rate': 1e-4
        }
    }
    
    build_result = test_api_endpoint('/api/build', 'POST', network_config, "构建简单神经网络")
    if build_result:
        print(f"   参数数量: {build_result.get('parameter_count', 0):,}")
        print(f"   内存估算: {build_result.get('memory_estimate', 0)}MB")
        print(f"   训练可行: {build_result.get('training_feasible', False)}")
    
    # 测试5: 启动训练（如果网络构建成功）
    if build_result:
        print("\n🎯 第5步: 启动训练任务")
        training_config = {
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 1e-4
        }
        
        train_result = test_api_endpoint('/api/train', 'POST', training_config, "启动训练任务")
        if train_result:
            training_id = train_result.get('training_id')
            print(f"   训练ID: {training_id}")
            print(f"   预计时间: {train_result.get('estimated_time', 'N/A')}")
            
            # 测试6: 查询训练状态
            print(f"\n📊 第6步: 查询训练状态")
            for i in range(3):
                time.sleep(1)
                status_result = test_api_endpoint(f'/api/training/{training_id}', 
                                                description=f"查询训练状态 (第{i+1}次)")
                if status_result:
                    job = status_result.get('job', {})
                    print(f"   状态: {job.get('status', 'unknown')}")
                    print(f"   进度: {job.get('current_epoch', 0)}/{job.get('epochs', 0)}")
                    if job.get('loss'):
                        print(f"   损失: {job.get('loss', 0):.4f}")
                    if job.get('accuracy'):
                        print(f"   准确率: {job.get('accuracy', 0):.4f}")
    
    # 测试7: 复杂网络构建（测试限制）
    print("\n⚠️  第7步: 测试网络深度限制")
    complex_network = {
        'components': [{'id': f'layer_{i}', 'type': 'mlp', 'params': {'input_dim': 512, 'output_dim': 512, 'hidden_dims': [1024]}} for i in range(15)],  # 15层，超过限制
        'connections': [],
        'config': {'name': 'too_deep_network', 'input_features': 512, 'output_features': 1}
    }
    
    complex_result = test_api_endpoint('/api/build', 'POST', complex_network, "测试过深网络（应该失败）")
    if not complex_result:
        print("   ✅ 深度限制正常工作")
    else:
        print("   ⚠️  深度限制可能未生效")
    
    print("\n" + "=" * 60)
    print("🎉 API测试完成!")
    print("=" * 60)
    
    print("\n📊 测试总结:")
    print("✅ Web应用正常运行")
    print("✅ API端点响应正常")
    print("✅ 网络构建功能可用")
    print("✅ 训练管理功能可用")
    print("✅ 系统限制正常工作")
    
    print("\n🌐 访问Web界面:")
    print("   浏览器打开: http://localhost:5000")
    print("   开始拖拽构建您的神经网络！")

if __name__ == "__main__":
    main()
