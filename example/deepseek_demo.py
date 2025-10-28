#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeek-V3-AWQ 模型交互演示
使用OpenAI接口格式与自定义模型端点交互
"""

import openai
import json
from typing import List, Dict, Any
import time

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: API密钥
            base_url: 模型服务地址
            model_name: 模型名称
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        进行聊天对话
        
        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "消息内容"}]
            **kwargs: 其他参数（temperature, max_tokens等）
            
        Returns:
            模型的回复内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用模型时出现错误: {e}")
            return None
    
    def stream_chat_completion(self, messages: List[Dict[str, str]], **kwargs):
        """
        流式聊天对话
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            流式返回的内容片段
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"流式调用模型时出现错误: {e}")
            yield None

def demo_basic_chat():
    """基础聊天演示"""
    print("=" * 50)
    print("基础聊天演示")
    print("=" * 50)
    
    # 初始化客户端
    client = DeepSeekClient(
        api_key="e3a1c7d92f6b48e09f2cb943da8e7c4f64bb1d3ab2749f05f1a0bde25dce9f3a",
        base_url="http://deepseek01.huafeng.com:8080/v1",
        model_name="deepseek-v3-awq"
    )
    
    # 准备消息
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手，请用中文回答问题。"},
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ]
    
    # 调用模型
    print("发送消息:", messages[-1]["content"])
    print("-" * 30)
    
    response = client.chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=1024
    )
    
    if response:
        print("模型回复:")
        print(response)
    else:
        print("调用失败")

def demo_stream_chat():
    """流式聊天演示"""
    print("\n" + "=" * 50)
    print("流式聊天演示")
    print("=" * 50)
    
    # 初始化客户端
    client = DeepSeekClient(
        api_key="e3a1c7d92f6b48e09f2cb943da8e7c4f64bb1d3ab2749f05f1a0bde25dce9f3a",
        base_url="http://deepseek01.huafeng.com:8080/v1",
        model_name="deepseek-v3-awq"
    )
    
    # 准备消息
    messages = [
        {"role": "system", "content": "你是一个专业的Python编程助手。"},
        {"role": "user", "content": "请写一个Python函数来计算斐波那契数列的前n项。"}
    ]
    
    print("发送消息:", messages[-1]["content"])
    print("-" * 30)
    print("模型回复（流式）:")
    
    # 流式调用
    full_response = ""
    for chunk in client.stream_chat_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=1024
    ):
        if chunk:
            print(chunk, end="", flush=True)
            full_response += chunk
    
    print("\n")
    print(f"完整回复长度: {len(full_response)} 字符")

def demo_conversation():
    """多轮对话演示"""
    print("\n" + "=" * 50)
    print("多轮对话演示")
    print("=" * 50)
    
    # 初始化客户端
    client = DeepSeekClient(
        api_key="e3a1c7d92f6b48e09f2cb943da8e7c4f64bb1d3ab2749f05f1a0bde25dce9f3a",
        base_url="http://deepseek01.huafeng.com:8080/v1",
        model_name="deepseek-v3-awq"
    )
    
    # 初始化对话历史
    conversation_history = [
        {"role": "system", "content": "你是一个专业的数据科学助手，擅长机器学习和数据分析。"}
    ]
    
    # 模拟多轮对话
    questions = [
        "什么是机器学习？",
        "监督学习和无监督学习有什么区别？",
        "能给我推荐一些常用的Python机器学习库吗？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n第{i}轮对话:")
        print(f"用户: {question}")
        
        # 添加用户消息到历史
        conversation_history.append({"role": "user", "content": question})
        
        # 获取模型回复
        response = client.chat_completion(
            messages=conversation_history,
            temperature=0.7,
            max_tokens=512
        )
        
        if response:
            print(f"助手: {response}")
            # 添加助手回复到历史
            conversation_history.append({"role": "assistant", "content": response})
        else:
            print("调用失败")
            break
        
        time.sleep(1)  # 避免请求过于频繁

def demo_different_parameters():
    """不同参数设置演示"""
    print("\n" + "=" * 50)
    print("不同参数设置演示")
    print("=" * 50)
    
    # 初始化客户端
    client = DeepSeekClient(
        api_key="e3a1c7d92f6b48e09f2cb943da8e7c4f64bb1d3ab2749f05f1a0bde25dce9f3a",
        base_url="http://deepseek01.huafeng.com:8080/v1",
        model_name="deepseek-v3-awq"
    )
    
    # 测试消息
    messages = [
        {"role": "system", "content": "你是一个创意写作助手。"},
        {"role": "user", "content": "写一个关于AI和人类合作的短故事。"}
    ]
    
    # 不同的参数设置
    parameter_sets = [
        {"name": "保守设置", "temperature": 0.3, "max_tokens": 200},
        {"name": "平衡设置", "temperature": 0.7, "max_tokens": 300},
        {"name": "创意设置", "temperature": 1.0, "max_tokens": 300}
    ]
    
    for param_set in parameter_sets:
        print(f"\n{param_set['name']} (temperature={param_set['temperature']}):")
        print("-" * 30)
        
        response = client.chat_completion(
            messages=messages,
            temperature=param_set["temperature"],
            max_tokens=param_set["max_tokens"]
        )
        
        if response:
            print(response[:200] + "..." if len(response) > 200 else response)
        else:
            print("调用失败")
        
        time.sleep(1)

def interactive_chat():
    """交互式聊天模式"""
    print("\n" + "=" * 50)
    print("交互式聊天模式")
    print("输入 'quit' 或 '退出' 来结束对话")
    print("=" * 50)
    
    # 初始化客户端
    client = DeepSeekClient(
        api_key="e3a1c7d92f6b48e09f2cb943da8e7c4f64bb1d3ab2749f05f1a0bde25dce9f3a",
        base_url="http://deepseek01.huafeng.com:8080/v1",
        model_name="deepseek-v3-awq"
    )
    
    # 初始化对话历史
    conversation_history = [
        {"role": "system", "content": "你是一个友好的AI助手，请用中文与用户对话。"}
    ]
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            # 添加用户消息
            conversation_history.append({"role": "user", "content": user_input})
            
            print("AI助手: ", end="", flush=True)
            
            # 流式获取回复
            full_response = ""
            for chunk in client.stream_chat_completion(
                messages=conversation_history,
                temperature=0.7,
                max_tokens=1024
            ):
                if chunk:
                    print(chunk, end="", flush=True)
                    full_response += chunk
            
            print()  # 换行
            
            # 添加助手回复到历史
            if full_response:
                conversation_history.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            print("\n\n对话被中断，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")

def main():
    """主函数"""
    print("DeepSeek-V3-AWQ 模型交互演示")
    print("=" * 50)
    
    try:
        # 运行各种演示
        demo_basic_chat()
        demo_stream_chat()
        demo_conversation()
        demo_different_parameters()
        
        # 询问是否进入交互模式
        print("\n" + "=" * 50)
        choice = input("是否进入交互式聊天模式？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是', '好']:
            interactive_chat()
        
    except Exception as e:
        print(f"程序运行时出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 