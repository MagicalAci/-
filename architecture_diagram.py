#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiFX AI搜索系统 - 产品全景架构图生成器
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ArchitectureDiagramGenerator:
    """架构图生成器"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.colors = {
            'frontend': '#3498db',
            'ai_service': '#e74c3c', 
            'search_engine': '#f39c12',
            'data_layer': '#27ae60',
            'connection': '#95a5a6',
            'text': '#2c3e50'
        }
    
    def create_system_architecture(self):
        """创建系统架构图"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 12))
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(0, 12)
        self.ax.axis('off')
        
        # 标题
        self.ax.text(8, 11.5, 'WikiFX AI搜索系统全景架构图', 
                    fontsize=20, fontweight='bold', ha='center', color=self.colors['text'])
        
        # 前端界面层
        self._draw_frontend_layer()
        
        # AI服务层
        self._draw_ai_service_layer()
        
        # 搜索引擎层
        self._draw_search_engine_layer()
        
        # 数据层
        self._draw_data_layer()
        
        # 绘制连接线
        self._draw_connections()
        
        # 添加图例
        self._add_legend()
        
        plt.tight_layout()
        return self.fig
    
    def _draw_frontend_layer(self):
        """绘制前端界面层"""
        # 主框架
        frontend_box = FancyBboxPatch(
            (1, 9.5), 14, 1.2,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['frontend'],
            edgecolor='black',
            alpha=0.8
        )
        self.ax.add_patch(frontend_box)
        
        # 标题
        self.ax.text(8, 10.3, '前端界面层', fontsize=14, fontweight='bold', 
                    ha='center', va='center', color='white')
        
        # 子组件
        components = [
            (2.5, '搜索输入框'),
            (5.5, '智能提示补全'),
            (8.5, '结果展示界面'),
            (11.5, '个性化推荐'),
            (13.5, '语音输入')
        ]
        
        for x, name in components:
            component_box = Rectangle((x-0.7, 9.6), 1.4, 0.4, 
                                    facecolor='white', edgecolor='black', alpha=0.9)
            self.ax.add_patch(component_box)
            self.ax.text(x, 9.8, name, fontsize=8, ha='center', va='center')
    
    def _draw_ai_service_layer(self):
        """绘制AI服务层"""
        # 主框架
        ai_box = FancyBboxPatch(
            (1, 7.5), 14, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['ai_service'],
            edgecolor='black',
            alpha=0.8
        )
        self.ax.add_patch(ai_box)
        
        # 标题
        self.ax.text(8, 8.7, 'AI服务层', fontsize=14, fontweight='bold', 
                    ha='center', va='center', color='white')
        
        # AI服务组件
        ai_components = [
            (2.5, 8.2, '意图识别\n引擎'),
            (5, 8.2, '实体抽取\n引擎'),
            (7.5, 8.2, '导航代理\n系统'),
            (10, 8.2, '个性化推荐\n引擎'),
            (12.5, 8.2, '结果生成器'),
            (6, 7.7, '对话管理器'),
            (9, 7.7, 'AI Agent\n协调器')
        ]
        
        for x, y, name in ai_components:
            component_box = FancyBboxPatch(
                (x-0.6, y-0.25), 1.2, 0.5,
                boxstyle="round,pad=0.05",
                facecolor='white',
                edgecolor='black',
                alpha=0.9
            )
            self.ax.add_patch(component_box)
            self.ax.text(x, y, name, fontsize=7, ha='center', va='center')
    
    def _draw_search_engine_layer(self):
        """绘制搜索引擎层"""
        # 主框架
        search_box = FancyBboxPatch(
            (1, 5.5), 14, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['search_engine'],
            edgecolor='black',
            alpha=0.8
        )
        self.ax.add_patch(search_box)
        
        # 标题
        self.ax.text(8, 6.7, '搜索引擎层', fontsize=14, fontweight='bold', 
                    ha='center', va='center', color='white')
        
        # 搜索引擎组件
        search_components = [
            (2.5, 6.2, '全文搜索\n引擎'),
            (5, 6.2, '语义搜索\n引擎'),
            (7.5, 6.2, '实时数据\n引擎'),
            (10, 6.2, '多媒体搜索\n引擎'),
            (12.5, 6.2, '知识图谱\n引擎'),
            (8, 5.7, '聚合搜索协调器')
        ]
        
        for x, y, name in search_components:
            component_box = FancyBboxPatch(
                (x-0.6, y-0.25), 1.2, 0.5,
                boxstyle="round,pad=0.05",
                facecolor='white',
                edgecolor='black',
                alpha=0.9
            )
            self.ax.add_patch(component_box)
            self.ax.text(x, y, name, fontsize=7, ha='center', va='center')
    
    def _draw_data_layer(self):
        """绘制数据层"""
        # 主框架
        data_box = FancyBboxPatch(
            (1, 2.5), 14, 2.5,
            boxstyle="round,pad=0.1",
            facecolor=self.colors['data_layer'],
            edgecolor='black',
            alpha=0.8
        )
        self.ax.add_patch(data_box)
        
        # 标题
        self.ax.text(8, 4.7, '数据层', fontsize=14, fontweight='bold', 
                    ha='center', va='center', color='white')
        
        # 数据库组件
        data_components = [
            # 第一行
            (3, 4.2, '交易商\n数据库'),
            (5.5, 4.2, '监管信息库'),
            (8, 4.2, '用户行为\n数据'),
            (10.5, 4.2, '内容知识库'),
            (13, 4.2, '实时行情\n数据'),
            # 第二行
            (3, 3.5, '用户评价库'),
            (5.5, 3.5, '风险评级库'),
            (8, 3.5, '向量数据库'),
            (10.5, 3.5, '新闻资讯库'),
            (13, 3.5, '教育内容库'),
            # 第三行
            (4.25, 2.8, '缓存层 (Redis)'),
            (8, 2.8, '消息队列 (RabbitMQ)'),
            (11.75, 2.8, '日志系统')
        ]
        
        for x, y, name in data_components:
            component_box = FancyBboxPatch(
                (x-0.7, y-0.25), 1.4, 0.5,
                boxstyle="round,pad=0.05",
                facecolor='white',
                edgecolor='black',
                alpha=0.9
            )
            self.ax.add_patch(component_box)
            self.ax.text(x, y, name, fontsize=7, ha='center', va='center')
    
    def _draw_connections(self):
        """绘制连接线"""
        # 垂直连接线
        connection_lines = [
            # 前端到AI服务层
            ((8, 9.5), (8, 9.0)),
            # AI服务层到搜索引擎层
            ((8, 7.5), (8, 7.0)),
            # 搜索引擎层到数据层
            ((8, 5.5), (8, 5.0))
        ]
        
        for start, end in connection_lines:
            self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                        color=self.colors['connection'], linewidth=3, alpha=0.7)
            # 添加箭头
            self.ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=self.colors['connection'], lw=2))
        
        # 数据流标签
        self.ax.text(8.5, 9.25, '用户查询', fontsize=8, ha='left', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        self.ax.text(8.5, 7.25, 'AI处理', fontsize=8, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        self.ax.text(8.5, 5.25, '数据检索', fontsize=8, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _add_legend(self):
        """添加图例"""
        legend_x = 1
        legend_y = 1.5
        
        # 图例框
        legend_box = FancyBboxPatch(
            (legend_x, legend_y-0.3), 14, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='#ecf0f1',
            edgecolor='black',
            alpha=0.9
        )
        self.ax.add_patch(legend_box)
        
        self.ax.text(legend_x + 7, legend_y + 0.7, '系统特性说明', 
                    fontsize=12, fontweight='bold', ha='center', va='center')
        
        features = [
            '• 智能意图识别：理解用户自然语言查询意图',
            '• 多模态搜索：支持文本、语音、图像等多种输入方式',
            '• 实时响应：< 500ms 搜索响应时间',
            '• 个性化推荐：基于用户行为的智能推荐系统',
            '• 高可用性：99.9% 系统可用性保障'
        ]
        
        for i, feature in enumerate(features):
            self.ax.text(legend_x + 0.5, legend_y + 0.3 - i*0.15, feature, 
                        fontsize=9, ha='left', va='center')
    
    def create_workflow_diagram(self):
        """创建工作流程图"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 10))
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')
        
        # 标题
        self.ax.text(8, 9.5, 'WikiFX AI搜索工作流程图', 
                    fontsize=20, fontweight='bold', ha='center', color=self.colors['text'])
        
        # 工作流步骤
        workflow_steps = [
            (2, 8, '用户输入\n自然语言查询', '#3498db'),
            (5, 8, '预处理\n文本清洗', '#9b59b6'),
            (8, 8, '意图识别\nAI分析', '#e74c3c'),
            (11, 8, '实体抽取\n关键信息', '#f39c12'),
            (14, 8, '上下文理解\n多轮对话', '#1abc9c'),
            
            (2, 5.5, '搜索策略\n智能选择', '#34495e'),
            (5, 5.5, '多源检索\n并行搜索', '#27ae60'),
            (8, 5.5, '结果聚合\n智能排序', '#e67e22'),
            (11, 5.5, '答案生成\nAI总结', '#8e44ad'),
            (14, 5.5, '结果展示\n用户界面', '#2980b9'),
            
            (4, 3, '反馈学习\n模型优化', '#c0392b'),
            (8, 3, '个性化更新\n用户画像', '#d35400'),
            (12, 3, '性能监控\n系统调优', '#7f8c8d')
        ]
        
        # 绘制步骤节点
        for x, y, text, color in workflow_steps:
            # 节点圆圈
            circle = Circle((x, y), 0.8, facecolor=color, edgecolor='black', alpha=0.8)
            self.ax.add_patch(circle)
            
            # 节点文字
            self.ax.text(x, y, text, fontsize=8, ha='center', va='center', 
                        color='white', fontweight='bold')
        
        # 绘制流程箭头
        arrows = [
            # 第一行流程
            ((2.8, 8), (4.2, 8)),
            ((5.8, 8), (7.2, 8)),
            ((8.8, 8), (10.2, 8)),
            ((11.8, 8), (13.2, 8)),
            # 向下转折
            ((14, 7.2), (14, 6.3)),
            ((13.2, 5.5), (11.8, 5.5)),
            ((10.2, 5.5), (8.8, 5.5)),
            ((7.2, 5.5), (5.8, 5.5)),
            ((4.2, 5.5), (2.8, 5.5)),
            # 反馈环路
            ((2, 4.7), (3.2, 3.8)),
            ((4.8, 3), (7.2, 3)),
            ((8.8, 3), (11.2, 3))
        ]
        
        for start, end in arrows:
            self.ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
        
        # 添加阶段标签
        self.ax.text(8, 8.8, '输入理解阶段', fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#ecf0f1', alpha=0.8))
        self.ax.text(8, 6.3, '搜索执行阶段', fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#ecf0f1', alpha=0.8))
        self.ax.text(8, 3.8, '学习优化阶段', fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#ecf0f1', alpha=0.8))
        
        plt.tight_layout()
        return self.fig
    
    def create_implementation_roadmap(self):
        """创建实施路线图"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 10))
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')
        
        # 标题
        self.ax.text(8, 9.5, 'WikiFX AI搜索实施路线图', 
                    fontsize=20, fontweight='bold', ha='center', color=self.colors['text'])
        
        # 时间轴
        timeline_y = 7
        self.ax.plot([1, 15], [timeline_y, timeline_y], color='#2c3e50', linewidth=4)
        
        # 各阶段
        phases = [
            (3, '第一期\n意图识别-导航代理\n(3个月)', '#e74c3c', [
                '意图识别模型开发',
                '导航代理系统构建',
                '基础搜索逻辑实现',
                'Demo原型发布'
            ]),
            (8, '第二期\n语义搜索-个性化\n(4个月)', '#f39c12', [
                '语义搜索引擎集成',
                '个性化推荐系统',
                '用户画像构建',
                'A/B测试优化'
            ]),
            (13, '第三期\n多模态-高级功能\n(3个月)', '#27ae60', [
                '语音搜索支持',
                '图像搜索功能',
                '对话式搜索',
                '全功能上线'
            ])
        ]
        
        for x, phase_text, color, features in phases:
            # 时间节点
            circle = Circle((x, timeline_y), 0.3, facecolor=color, edgecolor='black')
            self.ax.add_patch(circle)
            
            # 阶段框
            phase_box = FancyBboxPatch(
                (x-1.5, timeline_y+0.5), 3, 1.5,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                alpha=0.8
            )
            self.ax.add_patch(phase_box)
            
            # 阶段标题
            self.ax.text(x, timeline_y+1.25, phase_text, fontsize=10, fontweight='bold',
                        ha='center', va='center', color='white')
            
            # 功能列表
            for i, feature in enumerate(features):
                feature_box = FancyBboxPatch(
                    (x-1.4, timeline_y-1.5-i*0.4), 2.8, 0.35,
                    boxstyle="round,pad=0.05",
                    facecolor='white',
                    edgecolor=color,
                    alpha=0.9
                )
                self.ax.add_patch(feature_box)
                self.ax.text(x, timeline_y-1.33-i*0.4, feature, fontsize=8,
                           ha='center', va='center', color=color)
        
        # 关键指标
        self.ax.text(8, 5, '核心目标指标', fontsize=14, fontweight='bold', ha='center', va='center')
        
        metrics = [
            '搜索准确率提升 > 30%',
            '用户搜索步骤减少 > 50%',
            '搜索响应时间 < 500ms',
            '用户满意度提升 > 25%',
            '系统可用性 > 99.9%'
        ]
        
        for i, metric in enumerate(metrics):
            self.ax.text(8, 4.5-i*0.3, f'• {metric}', fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#ecf0f1', alpha=0.8))
        
        plt.tight_layout()
        return self.fig
    
    def save_diagrams(self):
        """保存所有架构图"""
        # 系统架构图
        fig1 = self.create_system_architecture()
        fig1.savefig('/workspace/wikifx_system_architecture.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 工作流程图
        fig2 = self.create_workflow_diagram()
        fig2.savefig('/workspace/wikifx_workflow_diagram.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 实施路线图
        fig3 = self.create_implementation_roadmap()
        fig3.savefig('/workspace/wikifx_implementation_roadmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        print("✅ 所有架构图已生成完成:")
        print("   📊 wikifx_system_architecture.png - 系统架构图")
        print("   🔄 wikifx_workflow_diagram.png - 工作流程图")
        print("   🗓️ wikifx_implementation_roadmap.png - 实施路线图")


if __name__ == "__main__":
    generator = ArchitectureDiagramGenerator()
    generator.save_diagrams()