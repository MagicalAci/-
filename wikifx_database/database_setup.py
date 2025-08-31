#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiFX数据库设置脚本
创建数据库表结构和导入数据
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
import os

class WikiFXDatabase:
    def __init__(self, db_path='wikifx.db'):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"成功连接到数据库: {self.db_path}")
            return True
        except Exception as e:
            print(f"连接数据库失败: {e}")
            return False
    
    def create_tables(self):
        """创建数据库表"""
        if not self.conn:
            print("数据库未连接")
            return False
        
        try:
            cursor = self.conn.cursor()
            
            # 创建交易商主表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS brokers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    rating REAL,
                    regulation TEXT,
                    min_deposit TEXT,
                    spread TEXT,
                    leverage TEXT,
                    platform TEXT,
                    url TEXT,
                    description TEXT,
                    country TEXT,
                    established INTEGER,
                    account_types TEXT,
                    payment_methods TEXT,
                    customer_support TEXT,
                    languages TEXT,
                    commission TEXT,
                    swap_free TEXT,
                    demo_account TEXT,
                    mobile_trading TEXT,
                    social_trading TEXT,
                    copy_trading TEXT,
                    vps_service TEXT,
                    education TEXT,
                    market_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建监管机构表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    country TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建交易平台表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS platforms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    type TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建国家表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS countries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    region TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建交易商评分历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS broker_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker_id INTEGER,
                    rating REAL,
                    review_count INTEGER,
                    rating_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (broker_id) REFERENCES brokers (id)
                )
            ''')
            
            # 创建交易商评论表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS broker_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker_id INTEGER,
                    user_name TEXT,
                    rating INTEGER,
                    review_text TEXT,
                    review_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (broker_id) REFERENCES brokers (id)
                )
            ''')
            
            self.conn.commit()
            print("数据库表创建成功")
            return True
            
        except Exception as e:
            print(f"创建表失败: {e}")
            return False
    
    def insert_sample_data(self):
        """插入示例数据到辅助表"""
        if not self.conn:
            print("数据库未连接")
            return False
        
        try:
            cursor = self.conn.cursor()
            
            # 插入监管机构数据
            regulations = [
                ('FCA', '英国', '英国金融行为监管局'),
                ('ASIC', '澳大利亚', '澳大利亚证券投资委员会'),
                ('CySEC', '塞浦路斯', '塞浦路斯证券交易委员会'),
                ('FSCA', '南非', '南非金融部门行为监管局'),
                ('MAS', '新加坡', '新加坡金融管理局'),
                ('FSA', '日本', '日本金融厅'),
                ('BaFin', '德国', '德国联邦金融监管局'),
                ('CONSOB', '意大利', '意大利证券交易委员会'),
                ('CNMV', '西班牙', '西班牙国家证券市场委员会'),
                ('FINMA', '瑞士', '瑞士金融市场监管局'),
                ('MFSA', '马耳他', '马耳他金融服务局'),
                ('FMA', '新西兰', '新西兰金融市场管理局'),
                ('FSPR', '新西兰', '新西兰金融服务提供商注册处'),
                ('VFSC', '瓦努阿图', '瓦努阿图金融服务委员会'),
                ('FSC', '毛里求斯', '毛里求斯金融服务委员会'),
                ('CIMA', '开曼群岛', '开曼群岛金融管理局'),
                ('BVI FSC', '英属维尔京群岛', '英属维尔京群岛金融服务委员会')
            ]
            
            cursor.executemany('''
                INSERT OR IGNORE INTO regulations (name, country, description)
                VALUES (?, ?, ?)
            ''', regulations)
            
            # 插入交易平台数据
            platforms = [
                ('MT4', 'MetaTrader', 'MetaTrader 4交易平台'),
                ('MT5', 'MetaTrader', 'MetaTrader 5交易平台'),
                ('cTrader', 'cTrader', 'cTrader交易平台'),
                ('TradingView', 'TradingView', 'TradingView图表平台'),
                ('Proprietary', '自研', '自研交易平台'),
                ('WebTrader', '网页', '网页版交易平台'),
                ('Mobile App', '移动', '移动端交易应用'),
                ('MetaTrader', 'MetaTrader', 'MetaTrader系列平台'),
                ('NinjaTrader', 'NinjaTrader', 'NinjaTrader交易平台'),
                ('Trading Station', 'FXCM', 'FXCM交易站平台')
            ]
            
            cursor.executemany('''
                INSERT OR IGNORE INTO platforms (name, type, description)
                VALUES (?, ?, ?)
            ''', platforms)
            
            # 插入国家数据
            countries = [
                ('英国', '欧洲'),
                ('澳大利亚', '大洋洲'),
                ('塞浦路斯', '欧洲'),
                ('南非', '非洲'),
                ('新加坡', '亚洲'),
                ('日本', '亚洲'),
                ('德国', '欧洲'),
                ('意大利', '欧洲'),
                ('西班牙', '欧洲'),
                ('瑞士', '欧洲'),
                ('马耳他', '欧洲'),
                ('新西兰', '大洋洲'),
                ('瓦努阿图', '大洋洲'),
                ('塞舌尔', '非洲'),
                ('开曼群岛', '美洲'),
                ('英属维尔京群岛', '美洲'),
                ('美国', '美洲'),
                ('加拿大', '美洲'),
                ('香港', '亚洲'),
                ('台湾', '亚洲'),
                ('韩国', '亚洲'),
                ('马来西亚', '亚洲'),
                ('泰国', '亚洲'),
                ('印度尼西亚', '亚洲')
            ]
            
            cursor.executemany('''
                INSERT OR IGNORE INTO countries (name, region)
                VALUES (?, ?)
            ''', countries)
            
            self.conn.commit()
            print("示例数据插入成功")
            return True
            
        except Exception as e:
            print(f"插入示例数据失败: {e}")
            return False
    
    def import_csv_data(self, csv_file='wikifx_brokers.csv'):
        """从CSV文件导入交易商数据"""
        if not self.conn:
            print("数据库未连接")
            return False
        
        if not os.path.exists(csv_file):
            print(f"CSV文件不存在: {csv_file}")
            return False
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            print(f"读取到 {len(df)} 条交易商数据")
            
            # 插入数据到数据库
            df.to_sql('brokers', self.conn, if_exists='replace', index=False)
            
            print("CSV数据导入成功")
            return True
            
        except Exception as e:
            print(f"导入CSV数据失败: {e}")
            return False
    
    def create_indexes(self):
        """创建数据库索引"""
        if not self.conn:
            print("数据库未连接")
            return False
        
        try:
            cursor = self.conn.cursor()
            
            # 创建索引
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_brokers_name ON brokers(name)',
                'CREATE INDEX IF NOT EXISTS idx_brokers_rating ON brokers(rating)',
                'CREATE INDEX IF NOT EXISTS idx_brokers_regulation ON brokers(regulation)',
                'CREATE INDEX IF NOT EXISTS idx_brokers_country ON brokers(country)',
                'CREATE INDEX IF NOT EXISTS idx_brokers_platform ON brokers(platform)',
                'CREATE INDEX IF NOT EXISTS idx_broker_ratings_broker_id ON broker_ratings(broker_id)',
                'CREATE INDEX IF NOT EXISTS idx_broker_reviews_broker_id ON broker_reviews(broker_id)'
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            self.conn.commit()
            print("数据库索引创建成功")
            return True
            
        except Exception as e:
            print(f"创建索引失败: {e}")
            return False
    
    def get_database_stats(self):
        """获取数据库统计信息"""
        if not self.conn:
            print("数据库未连接")
            return {}
        
        try:
            cursor = self.conn.cursor()
            stats = {}
            
            # 获取各表的记录数
            tables = ['brokers', 'regulations', 'platforms', 'countries', 'broker_ratings', 'broker_reviews']
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = cursor.fetchone()[0]
                stats[table] = count
            
            # 获取交易商评分统计
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_brokers,
                    AVG(rating) as avg_rating,
                    MIN(rating) as min_rating,
                    MAX(rating) as max_rating
                FROM brokers 
                WHERE rating IS NOT NULL
            ''')
            rating_stats = cursor.fetchone()
            stats['rating_stats'] = {
                'total_brokers': rating_stats[0],
                'avg_rating': rating_stats[1],
                'min_rating': rating_stats[2],
                'max_rating': rating_stats[3]
            }
            
            return stats
            
        except Exception as e:
            print(f"获取数据库统计信息失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("数据库连接已关闭")
    
    def setup_database(self):
        """完整设置数据库"""
        print("开始设置WikiFX数据库...")
        
        if not self.connect():
            return False
        
        if not self.create_tables():
            return False
        
        if not self.insert_sample_data():
            return False
        
        if not self.import_csv_data():
            return False
        
        if not self.create_indexes():
            return False
        
        # 显示数据库统计信息
        stats = self.get_database_stats()
        print("\n数据库统计信息:")
        for table, count in stats.items():
            if table != 'rating_stats':
                print(f"  {table}: {count} 条记录")
        
        if 'rating_stats' in stats:
            rating_stats = stats['rating_stats']
            print(f"\n交易商评分统计:")
            print(f"  总交易商数: {rating_stats['total_brokers']}")
            print(f"  平均评分: {rating_stats['avg_rating']:.2f}")
            print(f"  最低评分: {rating_stats['min_rating']:.2f}")
            print(f"  最高评分: {rating_stats['max_rating']:.2f}")
        
        print("\n数据库设置完成!")
        return True

if __name__ == "__main__":
    db = WikiFXDatabase()
    db.setup_database()
    db.close()