import sys
import os
import datetime

# 确保能导入 opencontext 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from opencontext.config.global_config import GlobalConfig
from opencontext.storage.global_storage import get_storage

def check_storage():
    print("=== MineContext 记忆存储检查工具 ===")
    print(f"当前时间: {datetime.datetime.now()}")
    
    # 1. 初始化
    print("\n[1/3] 正在连接数据库...")
    try:
        # 加载配置
        GlobalConfig.get_instance()
        # 获取存储实例
        storage = get_storage()
        if not storage:
            print("❌ 错误：无法初始化存储实例。请检查 config/config.yaml 配置。")
            return
        print("✅ 数据库连接成功。")
    except Exception as e:
        print(f"❌ 初始化异常: {e}")
        return

    # 2. 检查向量数据库 (长期记忆的核心)
    print("\n[2/3] 正在读取长期记忆 (Vector DB)...")
    try:
        # 获取各类型的统计
        counts = storage.get_all_processed_context_counts()
        total_count = sum(counts.values())
        
        print(f"📊 当前记忆总量: {total_count} 条")
        for ctx_type, count in counts.items():
            if count > 0:
                print(f"   • {ctx_type}: {count} 条")
        
        if total_count == 0:
            print("⚠️  警告：数据库是空的，尚未保存任何记忆。")
        else:
            # 获取所有记忆并展平
            print("\n🔍 最近存入的 5 条记忆详情:")
            raw_data = storage.get_all_processed_contexts(limit=10) # 获取稍多一点以便排序
            
            all_contexts = []
            for type_list in raw_data.values():
                all_contexts.extend(type_list)
            
            # 按创建时间倒序排序（最新的在最前）
            all_contexts.sort(key=lambda x: x.properties.create_time, reverse=True)
            
            # 打印最新的 5 条
            for i, ctx in enumerate(all_contexts[:5]):
                # 尝试获取来源
                source = "unknown"
                if ctx.properties.raw_properties:
                    source = ctx.properties.raw_properties[0].source.value
                
                print("-" * 50)
                print(f"记忆 #{i+1}")
                print(f"📅 时间: {ctx.properties.create_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"🏷️ 类型: {ctx.extracted_data.context_type.value}")
                print(f"📡 来源: {source}")
                print(f"📝 标题: {ctx.extracted_data.title}")
                print(f"📄 摘要: {ctx.extracted_data.summary}")
                if ctx.extracted_data.entities:
                    entities = [e if isinstance(e, str) else e.get('name', str(e)) for e in ctx.extracted_data.entities]
                    print(f"🔗 实体: {', '.join(entities)}")
                print("-" * 50)

    except Exception as e:
        print(f"❌ 读取向量数据库失败: {e}")
        import traceback
        traceback.print_exc()

    # 3. 物理文件检查
    print("\n[3/3] 物理存储路径检查...")
    # 假设是默认配置
    chroma_path = "./persist/chromadb"
    if os.path.exists(chroma_path):
        print(f"✅ ChromaDB 文件夹存在: {os.path.abspath(chroma_path)}")
    else:
        print(f"⚠️  ChromaDB 文件夹未找到 (可能是第一次运行还未落盘): {chroma_path}")

if __name__ == "__main__":
    check_storage()