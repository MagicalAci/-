# WikiFX AI搜索功能技术实现方案

## 1. 技术架构概览

### 1.1 整体架构设计
```
┌─────────────────────────────────────────────────────────────┐
│                    前端界面层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  搜索界面   │  │  结果展示   │  │  交互反馈   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    API网关层                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 请求路由    │  │ 认证授权    │  │ 限流控制    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                   AI搜索引擎层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 意图识别    │  │ 知识检索    │  │ 推理分析    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │ 结果生成    │  │ 交互优化    │                          │
│  └─────────────┘  └─────────────┘                          │
├─────────────────────────────────────────────────────────────┤
│                    数据服务层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 向量数据库  │  │ 知识图谱    │  │ 关系数据库  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    基础设施层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 模型服务    │  │ 缓存系统    │  │ 监控告警    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 技术栈选择

#### 1.2.1 前端技术栈
- **框架**: React 18 + TypeScript
- **状态管理**: Redux Toolkit
- **UI组件库**: Ant Design / Material-UI
- **构建工具**: Vite
- **样式**: Tailwind CSS + CSS Modules

#### 1.2.2 后端技术栈
- **语言**: Python 3.9+
- **框架**: FastAPI
- **异步处理**: Celery + Redis
- **数据库**: PostgreSQL + Redis
- **向量数据库**: Pinecone / Weaviate

#### 1.2.3 AI/ML技术栈
- **深度学习框架**: PyTorch / TensorFlow
- **NLP模型**: Transformers (Hugging Face)
- **向量化**: Sentence Transformers
- **大语言模型**: OpenAI GPT-4 / 开源替代方案

## 2. 核心模块技术实现

### 2.1 意图识别模块

#### 2.1.1 模型架构
```python
# 意图识别模型实现
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class IntentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_intents, dropout=0.1):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 意图分类器
class IntentRecognitionService:
    def __init__(self):
        self.model = IntentClassifier('bert-base-chinese', num_intents=8)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.intent_labels = [
            'broker_query', 'comparison', 'recommendation', 
            'information', 'regulation', 'cost', 'review', 'other'
        ]
    
    def predict_intent(self, query: str) -> dict:
        inputs = self.tokenizer(
            query, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True, 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_intent = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_intent].item()
        
        return {
            'intent': self.intent_labels[predicted_intent],
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }
```

#### 2.1.2 实体抽取
```python
# 命名实体识别
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

class EntityExtractor:
    def __init__(self):
        # 加载中文NLP模型
        self.nlp = spacy.load("zh_core_web_sm")
        
        # 金融领域实体词典
        self.financial_entities = {
            'broker': ['XM', 'FXTM', 'IC Markets', 'Pepperstone', 'OANDA'],
            'regulator': ['FCA', 'ASIC', 'CySEC', 'NFA', 'CFTC'],
            'instrument': ['外汇', '股票', '加密货币', '期货', '期权'],
            'country': ['英国', '澳大利亚', '塞浦路斯', '美国', '新加坡']
        }
    
    def extract_entities(self, query: str) -> dict:
        doc = self.nlp(query)
        
        entities = {
            'brokers': [],
            'regulators': [],
            'instruments': [],
            'countries': [],
            'custom': []
        }
        
        # 提取预定义实体
        for token in doc:
            for entity_type, entity_list in self.financial_entities.items():
                if token.text in entity_list:
                    entities[f'{entity_type}s'].append({
                        'text': token.text,
                        'type': entity_type,
                        'position': token.i
                    })
        
        # 提取自定义实体
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE', 'MONEY']:
                entities['custom'].append({
                    'text': ent.text,
                    'type': ent.label_,
                    'position': ent.start
                })
        
        return entities
```

### 2.2 知识检索模块

#### 2.2.1 向量化检索
```python
# 向量检索服务
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

class VectorSearchService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        
    def build_index(self, documents: list):
        """构建向量索引"""
        self.documents = documents
        
        # 生成文档向量
        texts = [doc['content'] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积索引
        self.index.add(embeddings.astype('float32'))
        
    def search(self, query: str, top_k: int = 10) -> list:
        """向量搜索"""
        # 查询向量化
        query_embedding = self.model.encode([query])
        
        # 搜索相似文档
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # 返回结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'relevance': self._calculate_relevance(score)
                })
        
        return results
    
    def _calculate_relevance(self, score: float) -> str:
        """计算相关性等级"""
        if score > 0.8:
            return 'high'
        elif score > 0.6:
            return 'medium'
        else:
            return 'low'
```

#### 2.2.2 混合检索策略
```python
# 混合检索服务
from elasticsearch import Elasticsearch
import re

class HybridSearchService:
    def __init__(self):
        self.es = Elasticsearch(['localhost:9200'])
        self.vector_service = VectorSearchService()
        
    def hybrid_search(self, query: str, filters: dict = None) -> dict:
        """混合检索：关键词 + 向量 + 规则"""
        
        # 1. 关键词检索
        keyword_results = self._keyword_search(query, filters)
        
        # 2. 向量检索
        vector_results = self.vector_service.search(query)
        
        # 3. 规则检索
        rule_results = self._rule_based_search(query)
        
        # 4. 结果融合
        fused_results = self._fuse_results(
            keyword_results, 
            vector_results, 
            rule_results
        )
        
        return {
            'results': fused_results,
            'search_metadata': {
                'query': query,
                'filters': filters,
                'total_results': len(fused_results)
            }
        }
    
    def _keyword_search(self, query: str, filters: dict) -> list:
        """Elasticsearch关键词检索"""
        search_body = {
            'query': {
                'bool': {
                    'must': [
                        {
                            'multi_match': {
                                'query': query,
                                'fields': ['title^2', 'content', 'tags'],
                                'type': 'best_fields'
                            }
                        }
                    ],
                    'filter': []
                }
            },
            'highlight': {
                'fields': {
                    'content': {}
                }
            }
        }
        
        # 添加过滤条件
        if filters:
            for key, value in filters.items():
                search_body['query']['bool']['filter'].append({
                    'term': {key: value}
                })
        
        response = self.es.search(
            index='wikifx_content',
            body=search_body,
            size=20
        )
        
        return response['hits']['hits']
    
    def _rule_based_search(self, query: str) -> list:
        """基于规则的检索"""
        results = []
        
        # 监管机构匹配
        regulators = ['FCA', 'ASIC', 'CySEC', 'NFA']
        for regulator in regulators:
            if regulator.lower() in query.lower():
                results.extend(self._get_regulated_brokers(regulator))
        
        # 交易品种匹配
        instruments = ['外汇', '股票', '加密货币']
        for instrument in instruments:
            if instrument in query:
                results.extend(self._get_instrument_brokers(instrument))
        
        return results
    
    def _fuse_results(self, keyword_results, vector_results, rule_results):
        """结果融合算法"""
        # 使用加权融合策略
        fused = {}
        
        # 关键词结果权重
        for hit in keyword_results:
            doc_id = hit['_id']
            fused[doc_id] = {
                'document': hit['_source'],
                'score': hit['_score'] * 0.4,
                'source': 'keyword'
            }
        
        # 向量结果权重
        for result in vector_results:
            doc_id = result['document']['id']
            if doc_id in fused:
                fused[doc_id]['score'] += result['score'] * 0.4
                fused[doc_id]['source'] = 'hybrid'
            else:
                fused[doc_id] = {
                    'document': result['document'],
                    'score': result['score'] * 0.4,
                    'source': 'vector'
                }
        
        # 规则结果权重
        for result in rule_results:
            doc_id = result['id']
            if doc_id in fused:
                fused[doc_id]['score'] += 0.2
            else:
                fused[doc_id] = {
                    'document': result,
                    'score': 0.2,
                    'source': 'rule'
                }
        
        # 排序并返回
        sorted_results = sorted(
            fused.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        return sorted_results[:20]
```

### 2.3 推理分析模块

#### 2.3.1 大语言模型集成
```python
# LLM推理服务
import openai
from typing import List, Dict, Any

class LLMReasoningService:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def analyze_query(self, query: str, context: List[Dict]) -> Dict[str, Any]:
        """使用LLM分析查询和上下文"""
        
        # 构建提示词
        prompt = self._build_analysis_prompt(query, context)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一个专业的金融搜索分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            return self._parse_analysis(analysis)
            
        except Exception as e:
            return {
                'error': str(e),
                'fallback_analysis': self._fallback_analysis(query, context)
            }
    
    def _build_analysis_prompt(self, query: str, context: List[Dict]) -> str:
        """构建分析提示词"""
        context_text = "\n".join([
            f"- {doc['title']}: {doc['content'][:200]}..."
            for doc in context[:5]
        ])
        
        return f"""
        用户查询: {query}
        
        相关上下文:
        {context_text}
        
        请分析用户的查询意图，并提供以下信息:
        1. 主要搜索意图
        2. 关键实体和属性
        3. 推荐的操作建议
        4. 可能的后续问题
        
        请以JSON格式返回分析结果。
        """
    
    def _parse_analysis(self, analysis: str) -> Dict[str, Any]:
        """解析LLM分析结果"""
        try:
            import json
            return json.loads(analysis)
        except:
            return {
                'intent': 'unknown',
                'entities': [],
                'recommendations': [],
                'raw_analysis': analysis
            }
    
    def _fallback_analysis(self, query: str, context: List[Dict]) -> Dict[str, Any]:
        """备用分析逻辑"""
        return {
            'intent': 'information_query',
            'entities': [],
            'recommendations': ['查看相关文档', '联系客服'],
            'confidence': 0.5
        }
```

#### 2.3.2 知识图谱查询
```python
# 知识图谱服务
from neo4j import GraphDatabase
import networkx as nx

class KnowledgeGraphService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def query_related_entities(self, entity_name: str, relationship_type: str = None) -> List[Dict]:
        """查询相关实体"""
        with self.driver.session() as session:
            if relationship_type:
                query = """
                MATCH (e1)-[r:%s]->(e2)
                WHERE e1.name = $entity_name
                RETURN e2.name as name, e2.type as type, r.weight as weight
                ORDER BY r.weight DESC
                LIMIT 10
                """ % relationship_type
            else:
                query = """
                MATCH (e1)-[r]->(e2)
                WHERE e1.name = $entity_name
                RETURN e2.name as name, e2.type as type, type(r) as relationship, r.weight as weight
                ORDER BY r.weight DESC
                LIMIT 10
                """
            
            result = session.run(query, entity_name=entity_name)
            return [record.data() for record in result]
    
    def find_path_between_entities(self, entity1: str, entity2: str) -> List[Dict]:
        """查找两个实体之间的路径"""
        with self.driver.session() as session:
            query = """
            MATCH path = shortestPath((e1)-[*]-(e2))
            WHERE e1.name = $entity1 AND e2.name = $entity2
            RETURN path
            LIMIT 5
            """
            
            result = session.run(query, entity1=entity1, entity2=entity2)
            paths = []
            for record in result:
                path = record['path']
                path_data = []
                for node in path.nodes:
                    path_data.append({
                        'name': node['name'],
                        'type': node['type']
                    })
                paths.append(path_data)
            
            return paths
    
    def get_entity_attributes(self, entity_name: str) -> Dict:
        """获取实体属性"""
        with self.driver.session() as session:
            query = """
            MATCH (e)
            WHERE e.name = $entity_name
            RETURN e
            """
            
            result = session.run(query, entity_name=entity_name)
            record = result.single()
            if record:
                return dict(record['e'])
            return {}
```

### 2.4 结果生成模块

#### 2.4.1 模板化结果生成
```python
# 结果生成服务
from jinja2 import Template
import json

class ResultGenerationService:
    def __init__(self):
        self.templates = self._load_templates()
        
    def generate_search_results(self, intent: str, entities: List[Dict], 
                              search_results: List[Dict]) -> Dict[str, Any]:
        """生成搜索结果"""
        
        # 根据意图选择模板
        template = self.templates.get(intent, self.templates['default'])
        
        # 处理搜索结果
        processed_results = self._process_search_results(search_results)
        
        # 生成响应
        response = {
            'intent': intent,
            'entities': entities,
            'results': processed_results,
            'summary': self._generate_summary(intent, processed_results),
            'suggestions': self._generate_suggestions(intent, entities),
            'template': template['name']
        }
        
        return response
    
    def _load_templates(self) -> Dict[str, Dict]:
        """加载结果模板"""
        return {
            'broker_query': {
                'name': 'broker_list',
                'template': """
                <div class="broker-results">
                    <h2>找到 {{ results|length }} 个符合条件的交易商</h2>
                    {% for broker in results %}
                    <div class="broker-card">
                        <h3>{{ broker.name }}</h3>
                        <p>{{ broker.description }}</p>
                        <div class="broker-tags">
                            {% for tag in broker.tags %}
                            <span class="tag">{{ tag }}</span>
                            {% endfor %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
                """
            },
            'comparison': {
                'name': 'comparison_table',
                'template': """
                <div class="comparison-results">
                    <h2>交易商对比分析</h2>
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>对比项目</th>
                                {% for broker in results %}
                                <th>{{ broker.name }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            {% for metric in comparison_metrics %}
                            <tr>
                                <td>{{ metric.name }}</td>
                                {% for broker in results %}
                                <td>{{ broker[metric.key] }}</td>
                                {% endfor %}
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                """
            },
            'default': {
                'name': 'general_list',
                'template': """
                <div class="general-results">
                    <h2>搜索结果</h2>
                    {% for result in results %}
                    <div class="result-item">
                        <h3>{{ result.title }}</h3>
                        <p>{{ result.content }}</p>
                    </div>
                    {% endfor %}
                </div>
                """
            }
        }
    
    def _process_search_results(self, results: List[Dict]) -> List[Dict]:
        """处理搜索结果"""
        processed = []
        for result in results:
            processed_result = {
                'id': result.get('id'),
                'title': result.get('title', ''),
                'content': result.get('content', ''),
                'type': result.get('type', 'unknown'),
                'score': result.get('score', 0),
                'metadata': result.get('metadata', {})
            }
            
            # 根据类型添加特定字段
            if result.get('type') == 'broker':
                processed_result.update({
                    'name': result.get('name'),
                    'regulation': result.get('regulation'),
                    'instruments': result.get('instruments', []),
                    'rating': result.get('rating', 0)
                })
            
            processed.append(processed_result)
        
        return processed
    
    def _generate_summary(self, intent: str, results: List[Dict]) -> str:
        """生成结果摘要"""
        if intent == 'broker_query':
            return f"找到 {len(results)} 个符合条件的交易商"
        elif intent == 'comparison':
            return f"为您对比了 {len(results)} 个交易商的关键指标"
        else:
            return f"找到 {len(results)} 条相关信息"
    
    def _generate_suggestions(self, intent: str, entities: List[Dict]) -> List[str]:
        """生成后续建议"""
        suggestions = []
        
        if intent == 'broker_query':
            suggestions.extend([
                "查看详细的交易商评价",
                "比较不同交易商的手续费",
                "了解监管合规情况"
            ])
        elif intent == 'comparison':
            suggestions.extend([
                "查看用户真实评价",
                "了解开户流程",
                "咨询客服获取更多信息"
            ])
        
        return suggestions
```

## 3. API接口设计

### 3.1 搜索API
```python
# FastAPI搜索接口
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="WikiFX AI Search API")

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    page: int = 1
    page_size: int = 20

class SearchResponse(BaseModel):
    query: str
    intent: Dict[str, Any]
    entities: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    summary: str
    suggestions: List[str]
    pagination: Dict[str, Any]

@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """AI搜索接口"""
    try:
        # 1. 意图识别
        intent_service = IntentRecognitionService()
        intent_result = intent_service.predict_intent(request.query)
        
        # 2. 实体抽取
        entity_service = EntityExtractor()
        entities = entity_service.extract_entities(request.query)
        
        # 3. 知识检索
        search_service = HybridSearchService()
        search_results = search_service.hybrid_search(
            request.query, 
            request.filters
        )
        
        # 4. 推理分析
        reasoning_service = LLMReasoningService(api_key="your-api-key")
        analysis = reasoning_service.analyze_query(
            request.query, 
            search_results['results']
        )
        
        # 5. 结果生成
        generation_service = ResultGenerationService()
        response = generation_service.generate_search_results(
            intent_result['intent'],
            entities,
            search_results['results']
        )
        
        # 6. 分页处理
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
        paginated_results = response['results'][start_idx:end_idx]
        
        return SearchResponse(
            query=request.query,
            intent=intent_result,
            entities=entities,
            results=paginated_results,
            summary=response['summary'],
            suggestions=response['suggestions'],
            pagination={
                'page': request.page,
                'page_size': request.page_size,
                'total': len(response['results']),
                'total_pages': (len(response['results']) + request.page_size - 1) // request.page_size
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/suggestions")
async def get_suggestions(query: str = Query(..., min_length=1)):
    """搜索建议接口"""
    try:
        # 实现搜索建议逻辑
        suggestions = [
            f"{query} 交易商",
            f"{query} 手续费",
            f"{query} 评价",
            f"{query} 监管"
        ]
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/autocomplete")
async def autocomplete(term: str = Query(..., min_length=1)):
    """自动完成接口"""
    try:
        # 实现自动完成逻辑
        completions = [
            f"{term} Group",
            f"{term} Markets",
            f"{term} Trading"
        ]
        return {"completions": completions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3.2 数据模型
```python
# 数据模型定义
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Broker(Base):
    __tablename__ = 'brokers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    regulation = Column(JSON)  # 监管信息
    instruments = Column(JSON)  # 交易品种
    rating = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SearchLog(Base):
    __tablename__ = 'search_logs'
    
    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    intent = Column(String(100))
    entities = Column(JSON)
    results_count = Column(Integer, default=0)
    user_id = Column(Integer)
    session_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

class UserFeedback(Base):
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True)
    search_id = Column(Integer)
    result_id = Column(Integer)
    rating = Column(Integer)  # 1-5星评价
    feedback_type = Column(String(50))  # click, like, dislike
    user_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
```

## 4. 部署和运维

### 4.1 Docker部署
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-search-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/wikifx
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
      - elasticsearch
    volumes:
      - ./models:/app/models
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=wikifx
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  elasticsearch:
    image: elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  kibana:
    image: kibana:7.17.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  postgres_data:
  redis_data:
  elasticsearch_data:
```

### 4.2 监控和日志
```python
# 监控配置
import logging
from prometheus_client import Counter, Histogram, start_http_server
import time

# 指标定义
SEARCH_REQUESTS = Counter('search_requests_total', 'Total search requests')
SEARCH_DURATION = Histogram('search_duration_seconds', 'Search request duration')
SEARCH_ERRORS = Counter('search_errors_total', 'Total search errors')

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_search.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 中间件
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 搜索接口监控
@app.post("/api/v1/search")
async def search(request: SearchRequest):
    SEARCH_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # 搜索逻辑...
        result = await perform_search(request)
        SEARCH_DURATION.observe(time.time() - start_time)
        return result
    except Exception as e:
        SEARCH_ERRORS.inc()
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 5. 性能优化

### 5.1 缓存策略
```python
# Redis缓存服务
import redis
import json
import hashlib
from typing import Optional, Any

class CacheService:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1小时
        
    def get_cache_key(self, query: str, filters: dict = None) -> str:
        """生成缓存键"""
        cache_data = {
            'query': query,
            'filters': filters or {}
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return f"search:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存"""
        try:
            ttl = ttl or self.default_ttl
            return self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """批量删除缓存"""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return 0
```

### 5.2 异步处理
```python
# Celery异步任务
from celery import Celery
from celery.utils.log import get_task_logger

celery_app = Celery('ai_search')
celery_app.config_from_object('celeryconfig')

logger = get_task_logger(__name__)

@celery_app.task
def update_search_index():
    """更新搜索索引"""
    try:
        # 增量更新逻辑
        logger.info("Starting search index update")
        # ... 更新逻辑
        logger.info("Search index update completed")
    except Exception as e:
        logger.error(f"Index update error: {e}")
        raise

@celery_app.task
def train_intent_model():
    """训练意图识别模型"""
    try:
        logger.info("Starting intent model training")
        # ... 训练逻辑
        logger.info("Intent model training completed")
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise

@celery_app.task
def process_user_feedback(feedback_data: dict):
    """处理用户反馈"""
    try:
        logger.info(f"Processing feedback: {feedback_data}")
        # ... 反馈处理逻辑
        logger.info("Feedback processing completed")
    except Exception as e:
        logger.error(f"Feedback processing error: {e}")
        raise
```

## 6. 测试策略

### 6.1 单元测试
```python
# 测试用例
import pytest
from unittest.mock import Mock, patch
from app.services.intent_recognition import IntentRecognitionService

class TestIntentRecognition:
    
    def setup_method(self):
        self.service = IntentRecognitionService()
    
    @patch('torch.load')
    def test_predict_intent(self, mock_load):
        # 模拟模型加载
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        # 测试查询
        query = "我想找一个受FCA监管的外汇交易商"
        result = self.service.predict_intent(query)
        
        assert 'intent' in result
        assert 'confidence' in result
        assert result['confidence'] > 0.5
    
    def test_entity_extraction(self):
        query = "比较XM和FXTM的手续费"
        entities = self.service.extract_entities(query)
        
        assert 'XM' in [e['text'] for e in entities['brokers']]
        assert 'FXTM' in [e['text'] for e in entities['brokers']]
```

### 6.2 集成测试
```python
# API集成测试
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_search_api():
    response = client.post("/api/v1/search", json={
        "query": "受FCA监管的外汇交易商",
        "page": 1,
        "page_size": 10
    })
    
    assert response.status_code == 200
    data = response.json()
    assert 'intent' in data
    assert 'results' in data
    assert len(data['results']) > 0

def test_suggestions_api():
    response = client.get("/api/v1/suggestions?query=XM")
    
    assert response.status_code == 200
    data = response.json()
    assert 'suggestions' in data
    assert len(data['suggestions']) > 0
```

## 7. 总结

本技术实现方案提供了WikiFX AI搜索功能的完整技术架构，包括：

1. **模块化设计**：意图识别、知识检索、推理分析、结果生成等核心模块
2. **技术栈选择**：现代化的技术栈，确保性能和可扩展性
3. **API设计**：RESTful API接口，支持多种搜索场景
4. **部署方案**：Docker容器化部署，支持生产环境
5. **监控运维**：完整的监控、日志和性能优化方案
6. **测试策略**：单元测试和集成测试覆盖

该方案为WikiFX平台提供了强大的AI搜索能力，能够理解用户意图，提供精准的搜索结果，并支持持续优化和扩展。