# 多模型智能客服系统（LangChain RAG / DeepSeek Chat）

---

## 📌 项目简介
本仓库提供 **两种形态的聊天机器人**，底层实现思路保持一致，方便在不同场景之间平滑切换。

| 形态 | 主要技术 | 适用场景 | 特点 |
|------|----------|----------|------|
| **标准聊天模式** | FastAPI + DeepSeek Chat SDK + Gradio | 纯对话、问答、客服 | 架构轻量、无向量库依赖，部署即用 |
| **RAG 强化模式** | LangChain + Qdrant（内存版） + DeepSeek Chat SDK + LangSmith | 需要知识库检索、上下文增强 | 复用同一 UI，增加 Retriever 链路与向量检索，链路级 Trace 可视化 |

两种模式共享统一 UI、接口与模型管理；区别仅在于 **是否启用向量检索链路**。

---

## 🧱 系统架构图（RAG 流程）
```mermaid
flowchart TD
  A[用户提问] -->|HTTP| B(LangChain Router)
  B --> C{Retriever}
  C -->|向量检索| D[[Qdrant (In‑Mem)]]
  C -->|原文档| E[Doc Store]
  B -.-> F[Prompt Template]
  F --> G(LLM ⟶ DeepSeek Chat SDK / GPT‑2)
  G --> H[Answer]
  B --> I((LangSmith Trace))
  I -->|可视化| J[监控面板]
  H --> User((用户回复))
```
> **标准聊天模式** 省略 C/D/E/F 节点，仅保留 LLM 调用与 LangSmith 追踪。

---

## 🔧 技术栈

| 分层 | 主要组件 | 说明 |
|------|-----------|------|
| API 层 | **FastAPI** | 提供 REST / SSE 接口（WebSocket 规划中） |
| 对话编排 | **LangChain** | Prompt 管理、Retriever 链（RAG 模式专用） |
| 模型 | **DeepSeek Chat SDK / GPT‑2 (本地)** | 按需切换，DeepSeek 通过官方 SDK 调用 |
| 向量检索 | **Qdrant (内存运行)** | 当前运行在内存模式，持久化功能后续支持 |
| 前端 UI | **Gradio** | 聊天界面、消息复制、示例问题 |
| 可观测性 | **LangSmith** | Trace、Token Cost、链路耗时 |
| 部署 | **Docker（Compose 规划中）** | 一键启动 API + UI |

---

## 🗂️ 目录结构
```text
 ezo-robot/
 ├── src/
 │   ├── api/            # FastAPI 端点
 │   ├── model/          # 模型与链路模块
 │   ├── frontend/       # Gradio 前端
 │   ├── files/          # 文件处理
 │   └── utils/          # 通用工具
 ├── statics/            # 静态知识库文件
 └── tests/              # 测试代码
```

---

## 🚀 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 配置 Key
cp .env.example .env && vim .env
```
### 启动标准聊天模式（DeepSeek Chat）
```bash
uvicorn src.api.endpoints:app --reload  # 8000
python src/frontend/app.py              # 7860
```
### 启动 RAG 模式（LangChain + Qdrant）
```bash
# 待完善：docker-compose up -d  # 规划中
```
浏览器访问 UI: <http://localhost:7860>    API 文档: <http://localhost:8000/docs>

---

## ✨ 项目亮点
- **双模式统一 UI**：同一前端可无缝切换普通聊天 / RAG。
- **全链路可视**：LangSmith Trace，异常定位 <10 min。
- **模型热切换**：支持 DeepSeek ↔ OpenAI 即时切换。
- **异步流式输出**：SSE 提升前端体验（响应速度取决于模型推理时延）。

---

## 🗺️ 未来规划
1. **Qdrant 持久化**（WAL + RocksDB）
2. **WebSocket / gRPC 实时双工接口**
3. **Prompt Cache**（向量缓存，缩短首字节）
4. **Prometheus + Grafana 指标与告警**
5. **Docker Compose / Kubernetes 部署**
6. **多模态支持 & 插件扩展**

