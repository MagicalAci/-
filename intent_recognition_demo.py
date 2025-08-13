#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiFX AI搜索 - 意图识别引擎
第一期Demo实现：意图识别和导航代理
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SearchIntent(Enum):
    """搜索意图枚举"""
    BROKER_REGULATION = "监管查询"  # 查询交易商监管信息
    RISK_ASSESSMENT = "风险评估"   # 评估交易商风险
    BROKER_COMPARISON = "对比选择"  # 对比多个交易商
    PLATFORM_FEATURES = "平台功能"  # 查询平台功能特性
    EDUCATION_BASIC = "基础知识"    # 学习基础知识
    EDUCATION_STRATEGY = "策略技巧"  # 学习交易策略
    MARKET_NEWS = "市场资讯"       # 获取市场新闻
    MARKET_ANALYSIS = "市场分析"   # 获取市场分析
    USER_SUPPORT = "用户服务"      # 用户支持和咨询
    UNKNOWN = "未知意图"

@dataclass
class Entity:
    """实体类"""
    name: str
    type: str
    value: str
    confidence: float

@dataclass
class IntentResult:
    """意图识别结果"""
    intent: SearchIntent
    confidence: float
    entities: List[Entity]
    query_type: str
    suggested_actions: List[str]

class IntentRecognitionEngine:
    """意图识别引擎"""
    
    def __init__(self):
        self.intent_patterns = self._load_intent_patterns()
        self.entity_patterns = self._load_entity_patterns()
        self.broker_names = self._load_broker_names()
        
    def _load_intent_patterns(self) -> Dict[SearchIntent, List[str]]:
        """加载意图识别模式"""
        return {
            SearchIntent.BROKER_REGULATION: [
                r".*监管.*", r".*牌照.*", r".*合规.*", r".*正规.*",
                r".*FSA.*", r".*CySEC.*", r".*ASIC.*", r".*FCA.*"
            ],
            SearchIntent.RISK_ASSESSMENT: [
                r".*安全.*", r".*可靠.*", r".*风险.*", r".*信任.*",
                r".*怎么样.*", r".*评价.*", r".*口碑.*"
            ],
            SearchIntent.BROKER_COMPARISON: [
                r".*对比.*", r".*比较.*", r".*哪个.*", r".*选择.*",
                r".*vs.*", r".*和.*", r".*还是.*"
            ],
            SearchIntent.PLATFORM_FEATURES: [
                r".*手续费.*", r".*点差.*", r".*杠杆.*", r".*入金.*",
                r".*出金.*", r".*平台.*", r".*软件.*", r".*APP.*"
            ],
            SearchIntent.EDUCATION_BASIC: [
                r".*入门.*", r".*基础.*", r".*什么是.*", r".*如何.*",
                r".*教程.*", r".*学习.*"
            ],
            SearchIntent.EDUCATION_STRATEGY: [
                r".*策略.*", r".*技巧.*", r".*方法.*", r".*分析.*",
                r".*指标.*", r".*信号.*"
            ],
            SearchIntent.MARKET_NEWS: [
                r".*新闻.*", r".*消息.*", r".*事件.*", r".*政策.*",
                r".*发布.*", r".*公告.*"
            ],
            SearchIntent.MARKET_ANALYSIS: [
                r".*走势.*", r".*预测.*", r".*趋势.*", r".*观点.*",
                r".*分析师.*", r".*专家.*"
            ],
            SearchIntent.USER_SUPPORT: [
                r".*问题.*", r".*帮助.*", r".*客服.*", r".*投诉.*",
                r".*联系.*", r".*解决.*"
            ]
        }
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """加载实体识别模式"""
        return {
            "broker_name": [
                r"XM", r"IG", r"FXCM", r"OANDA", r"Interactive Brokers",
                r"嘉盛", r"福汇", r"安达", r"盈透", r"艾福瑞"
            ],
            "currency_pair": [
                r"EUR/USD", r"GBP/USD", r"USD/JPY", r"USD/CHF",
                r"欧美", r"镑美", r"美日", r"美瑞"
            ],
            "feature_type": [
                r"点差", r"杠杆", r"手续费", r"佣金", r"出入金",
                r"平台", r"软件", r"APP", r"客服"
            ],
            "regulation_body": [
                r"FCA", r"CySEC", r"ASIC", r"FSA", r"NFA",
                r"英国金融", r"塞浦路斯", r"澳洲", r"美国"
            ]
        }
    
    def _load_broker_names(self) -> List[str]:
        """加载交易商名称列表"""
        return [
            "XM", "IG Markets", "FXCM", "OANDA", "Interactive Brokers",
            "嘉盛集团", "福汇", "安达", "盈透证券", "艾福瑞", "IC Markets",
            "Exness", "Plus500", "eToro", "ATFX", "ThinkMarkets"
        ]
    
    def recognize_intent(self, query: str) -> IntentResult:
        """识别用户查询意图"""
        query = query.strip().lower()
        
        # 意图识别
        intent, intent_confidence = self._classify_intent(query)
        
        # 实体抽取
        entities = self._extract_entities(query)
        
        # 查询类型判断
        query_type = self._determine_query_type(intent, entities)
        
        # 生成建议操作
        suggested_actions = self._generate_suggestions(intent, entities)
        
        return IntentResult(
            intent=intent,
            confidence=intent_confidence,
            entities=entities,
            query_type=query_type,
            suggested_actions=suggested_actions
        )
    
    def _classify_intent(self, query: str) -> Tuple[SearchIntent, float]:
        """分类用户意图"""
        max_confidence = 0.0
        best_intent = SearchIntent.UNKNOWN
        
        for intent, patterns in self.intent_patterns.items():
            confidence = 0.0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    confidence += 0.3
            
            # 权重调整
            if confidence > 0:
                confidence = min(confidence, 0.95)
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_intent = intent
        
        # 如果没有匹配的模式，使用启发式规则
        if max_confidence == 0:
            best_intent, max_confidence = self._heuristic_classification(query)
        
        return best_intent, max_confidence
    
    def _heuristic_classification(self, query: str) -> Tuple[SearchIntent, float]:
        """启发式分类方法"""
        # 包含交易商名称 + 评价词汇 = 风险评估
        has_broker = any(broker.lower() in query for broker in self.broker_names)
        evaluation_words = ["怎么样", "好不好", "靠谱", "可信"]
        has_evaluation = any(word in query for word in evaluation_words)
        
        if has_broker and has_evaluation:
            return SearchIntent.RISK_ASSESSMENT, 0.7
        
        # 包含"学习"、"教"等词汇 = 教育类
        education_words = ["学", "教", "了解", "知识"]
        if any(word in query for word in education_words):
            return SearchIntent.EDUCATION_BASIC, 0.6
        
        # 包含货币对 = 市场分析
        currency_words = ["EUR", "USD", "GBP", "JPY", "欧美", "镑美"]
        if any(word in query for word in currency_words):
            return SearchIntent.MARKET_ANALYSIS, 0.6
        
        return SearchIntent.UNKNOWN, 0.1
    
    def _extract_entities(self, query: str) -> List[Entity]:
        """抽取实体信息"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entity = Entity(
                        name=match.group(),
                        type=entity_type,
                        value=match.group(),
                        confidence=0.9
                    )
                    entities.append(entity)
        
        return entities
    
    def _determine_query_type(self, intent: SearchIntent, entities: List[Entity]) -> str:
        """确定查询类型"""
        if intent in [SearchIntent.BROKER_REGULATION, SearchIntent.RISK_ASSESSMENT]:
            return "交易商查询"
        elif intent in [SearchIntent.EDUCATION_BASIC, SearchIntent.EDUCATION_STRATEGY]:
            return "教育内容"
        elif intent in [SearchIntent.MARKET_NEWS, SearchIntent.MARKET_ANALYSIS]:
            return "市场资讯"
        else:
            return "综合查询"
    
    def _generate_suggestions(self, intent: SearchIntent, entities: List[Entity]) -> List[str]:
        """生成建议操作"""
        suggestions = []
        
        if intent == SearchIntent.BROKER_REGULATION:
            suggestions = [
                "查询监管牌照信息",
                "检查监管机构认证",
                "查看合规状态历史"
            ]
        elif intent == SearchIntent.RISK_ASSESSMENT:
            suggestions = [
                "查看用户评价和评分",
                "分析风险指标",
                "对比同类交易商"
            ]
        elif intent == SearchIntent.BROKER_COMPARISON:
            suggestions = [
                "生成对比表格",
                "分析优劣势",
                "推荐最佳选择"
            ]
        elif intent == SearchIntent.EDUCATION_BASIC:
            suggestions = [
                "推荐入门教程",
                "提供基础知识",
                "安排学习路径"
            ]
        else:
            suggestions = [
                "提供相关信息",
                "推荐相关内容",
                "联系专业顾问"
            ]
        
        return suggestions


class NavigationAgent:
    """导航代理"""
    
    def __init__(self, intent_engine: IntentRecognitionEngine):
        self.intent_engine = intent_engine
        self.search_strategies = self._load_search_strategies()
    
    def _load_search_strategies(self) -> Dict[SearchIntent, Dict]:
        """加载搜索策略配置"""
        return {
            SearchIntent.BROKER_REGULATION: {
                "data_sources": ["监管数据库", "牌照信息库"],
                "ranking_weights": {"监管等级": 0.4, "牌照数量": 0.3, "历史记录": 0.3},
                "filters": ["监管状态", "牌照类型"],
                "result_format": "监管信息卡片"
            },
            SearchIntent.RISK_ASSESSMENT: {
                "data_sources": ["用户评价库", "风险评级库", "投诉记录库"],
                "ranking_weights": {"用户评分": 0.3, "风险等级": 0.4, "投诉处理": 0.3},
                "filters": ["评分范围", "风险等级"],
                "result_format": "风险评估报告"
            },
            SearchIntent.BROKER_COMPARISON: {
                "data_sources": ["交易商数据库", "特征对比库"],
                "ranking_weights": {"综合评分": 0.5, "功能匹配": 0.3, "成本效益": 0.2},
                "filters": ["功能类型", "成本范围"],
                "result_format": "对比表格"
            }
        }
    
    def route_search(self, query: str) -> Dict:
        """路由搜索请求"""
        # 意图识别
        intent_result = self.intent_engine.recognize_intent(query)
        
        # 生成搜索策略
        strategy = self._generate_search_strategy(intent_result)
        
        # 模拟搜索执行
        search_results = self._execute_search(strategy, intent_result)
        
        return {
            "query": query,
            "intent_result": intent_result,
            "search_strategy": strategy,
            "results": search_results,
            "navigation_path": self._generate_navigation_path(intent_result)
        }
    
    def _generate_search_strategy(self, intent_result: IntentResult) -> Dict:
        """生成搜索策略"""
        base_strategy = self.search_strategies.get(
            intent_result.intent, 
            {
                "data_sources": ["通用数据库"],
                "ranking_weights": {"相关性": 1.0},
                "filters": [],
                "result_format": "通用结果"
            }
        )
        
        # 根据实体信息调整策略
        strategy = base_strategy.copy()
        
        # 如果有特定交易商实体，调整数据源
        broker_entities = [e for e in intent_result.entities if e.type == "broker_name"]
        if broker_entities:
            strategy["filters"].append(f"交易商: {broker_entities[0].value}")
        
        return strategy
    
    def _execute_search(self, strategy: Dict, intent_result: IntentResult) -> List[Dict]:
        """执行搜索（模拟实现）"""
        # 这里是模拟的搜索结果
        mock_results = []
        
        if intent_result.intent == SearchIntent.BROKER_REGULATION:
            mock_results = [
                {
                    "type": "监管信息",
                    "title": "XM集团监管状态",
                    "content": "受FCA、CySEC、ASIC多重监管",
                    "score": 0.95,
                    "source": "监管数据库"
                }
            ]
        elif intent_result.intent == SearchIntent.RISK_ASSESSMENT:
            mock_results = [
                {
                    "type": "风险评估",
                    "title": "XM交易商风险评级",
                    "content": "综合风险等级：低风险 (AAA级)",
                    "score": 0.92,
                    "source": "风险评级库"
                }
            ]
        
        return mock_results
    
    def _generate_navigation_path(self, intent_result: IntentResult) -> List[str]:
        """生成导航路径"""
        if intent_result.intent == SearchIntent.BROKER_REGULATION:
            return ["首页", "交易商", "监管信息", "详细查询"]
        elif intent_result.intent == SearchIntent.RISK_ASSESSMENT:
            return ["首页", "交易商", "风险评估", "综合评级"]
        elif intent_result.intent == SearchIntent.EDUCATION_BASIC:
            return ["首页", "教育中心", "基础知识", "入门教程"]
        else:
            return ["首页", "搜索结果"]


def demo_test():
    """演示测试"""
    print("=== WikiFX AI搜索引擎 Demo ===\n")
    
    # 初始化引擎
    intent_engine = IntentRecognitionEngine()
    navigation_agent = NavigationAgent(intent_engine)
    
    # 测试查询
    test_queries = [
        "XM交易商怎么样，安全吗？",
        "我想了解外汇交易的基础知识",
        "帮我对比一下XM和IG Markets",
        "FCA监管的交易商有哪些？",
        "今天的EUR/USD走势如何？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"📝 测试查询 {i}: {query}")
        
        # 执行搜索路由
        result = navigation_agent.route_search(query)
        
        print(f"🎯 识别意图: {result['intent_result'].intent.value}")
        print(f"📊 置信度: {result['intent_result'].confidence:.2f}")
        print(f"🏷️ 提取实体: {[f'{e.name}({e.type})' for e in result['intent_result'].entities]}")
        print(f"📂 查询类型: {result['intent_result'].query_type}")
        print(f"🗺️ 导航路径: {' > '.join(result['navigation_path'])}")
        print(f"💡 建议操作: {', '.join(result['intent_result'].suggested_actions)}")
        print("-" * 60)


if __name__ == "__main__":
    demo_test()