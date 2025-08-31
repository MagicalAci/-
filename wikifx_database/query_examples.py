#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiFX数据库查询示例
演示各种查询功能
"""

import sqlite3
import pandas as pd
from datetime import datetime

class WikiFXQuery:
    def __init__(self, db_path='wikifx.db'):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            print(f"连接数据库失败: {e}")
            return False
    
    def get_top_rated_brokers(self, limit=10):
        """获取评分最高的交易商"""
        query = '''
            SELECT name, rating, regulation, country, platform, min_deposit, spread
            FROM brokers 
            WHERE rating IS NOT NULL
            ORDER BY rating DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, self.conn, params=[limit])
        return df
    
    def get_brokers_by_regulation(self, regulation):
        """根据监管机构查询交易商"""
        query = '''
            SELECT name, rating, country, platform, min_deposit, spread
            FROM brokers 
            WHERE regulation = ?
            ORDER BY rating DESC
        '''
        df = pd.read_sql_query(query, self.conn, params=[regulation])
        return df
    
    def get_brokers_by_country(self, country):
        """根据国家查询交易商"""
        query = '''
            SELECT name, rating, regulation, platform, min_deposit, spread
            FROM brokers 
            WHERE country = ?
            ORDER BY rating DESC
        '''
        df = pd.read_sql_query(query, self.conn, params=[country])
        return df
    
    def get_brokers_by_platform(self, platform):
        """根据交易平台查询交易商"""
        query = '''
            SELECT name, rating, regulation, country, min_deposit, spread
            FROM brokers 
            WHERE platform = ?
            ORDER BY rating DESC
        '''
        df = pd.read_sql_query(query, self.conn, params=[platform])
        return df
    
    def search_brokers(self, keyword):
        """搜索交易商"""
        query = '''
            SELECT name, rating, regulation, country, platform, min_deposit, spread
            FROM brokers 
            WHERE name LIKE ? OR description LIKE ?
            ORDER BY rating DESC
        '''
        search_term = f'%{keyword}%'
        df = pd.read_sql_query(query, self.conn, params=[search_term, search_term])
        return df
    
    def get_regulation_stats(self):
        """获取监管机构统计"""
        query = '''
            SELECT regulation, COUNT(*) as broker_count, AVG(rating) as avg_rating
            FROM brokers 
            WHERE regulation IS NOT NULL
            GROUP BY regulation
            ORDER BY broker_count DESC
        '''
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_country_stats(self):
        """获取国家统计"""
        query = '''
            SELECT country, COUNT(*) as broker_count, AVG(rating) as avg_rating
            FROM brokers 
            WHERE country IS NOT NULL
            GROUP BY country
            ORDER BY broker_count DESC
        '''
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_platform_stats(self):
        """获取交易平台统计"""
        query = '''
            SELECT platform, COUNT(*) as broker_count, AVG(rating) as avg_rating
            FROM brokers 
            WHERE platform IS NOT NULL
            GROUP BY platform
            ORDER BY broker_count DESC
        '''
        df = pd.read_sql_query(query, self.conn)
        return df
    
    def get_brokers_by_deposit_range(self, min_amount, max_amount):
        """根据最低存款范围查询交易商"""
        query = '''
            SELECT name, rating, regulation, country, platform, min_deposit, spread
            FROM brokers 
            WHERE CAST(REPLACE(REPLACE(min_deposit, '$', ''), ',', '') AS INTEGER) 
                  BETWEEN ? AND ?
            ORDER BY rating DESC
        '''
        df = pd.read_sql_query(query, self.conn, params=[min_amount, max_amount])
        return df
    
    def get_brokers_by_spread_range(self, min_spread, max_spread):
        """根据点差范围查询交易商"""
        query = '''
            SELECT name, rating, regulation, country, platform, min_deposit, spread
            FROM brokers 
            WHERE CAST(REPLACE(REPLACE(spread, ' pips', ''), ',', '') AS REAL) 
                  BETWEEN ? AND ?
            ORDER BY rating DESC
        '''
        df = pd.read_sql_query(query, self.conn, params=[min_spread, max_spread])
        return df
    
    def get_comprehensive_broker_info(self, broker_name):
        """获取交易商详细信息"""
        query = '''
            SELECT *
            FROM brokers 
            WHERE name = ?
        '''
        df = pd.read_sql_query(query, self.conn, params=[broker_name])
        return df
    
    def get_brokers_with_features(self, features):
        """根据特性查询交易商"""
        conditions = []
        params = []
        
        for feature in features:
            if feature == 'demo_account':
                conditions.append('demo_account = "是"')
            elif feature == 'swap_free':
                conditions.append('swap_free = "是"')
            elif feature == 'mobile_trading':
                conditions.append('mobile_trading = "是"')
            elif feature == 'social_trading':
                conditions.append('social_trading = "是"')
            elif feature == 'copy_trading':
                conditions.append('copy_trading = "是"')
            elif feature == 'vps_service':
                conditions.append('vps_service = "是"')
            elif feature == 'education':
                conditions.append('education = "是"')
            elif feature == 'market_analysis':
                conditions.append('market_analysis = "是"')
        
        if not conditions:
            return pd.DataFrame()
        
        where_clause = ' AND '.join(conditions)
        query = f'''
            SELECT name, rating, regulation, country, platform, min_deposit, spread
            FROM brokers 
            WHERE {where_clause}
            ORDER BY rating DESC
        '''
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

def main():
    """主函数 - 演示各种查询"""
    query = WikiFXQuery()
    
    if not query.connect():
        return
    
    print("=" * 60)
    print("WikiFX数据库查询示例")
    print("=" * 60)
    
    # 1. 获取评分最高的交易商
    print("\n1. 评分最高的10个交易商:")
    print("-" * 40)
    top_brokers = query.get_top_rated_brokers(10)
    print(top_brokers.to_string(index=False))
    
    # 2. 监管机构统计
    print("\n2. 监管机构统计:")
    print("-" * 40)
    reg_stats = query.get_regulation_stats()
    print(reg_stats.to_string(index=False))
    
    # 3. 国家统计
    print("\n3. 国家统计 (前10名):")
    print("-" * 40)
    country_stats = query.get_country_stats().head(10)
    print(country_stats.to_string(index=False))
    
    # 4. 交易平台统计
    print("\n4. 交易平台统计:")
    print("-" * 40)
    platform_stats = query.get_platform_stats()
    print(platform_stats.to_string(index=False))
    
    # 5. 根据监管机构查询
    print("\n5. FCA监管的交易商:")
    print("-" * 40)
    fca_brokers = query.get_brokers_by_regulation('FCA')
    print(fca_brokers.to_string(index=False))
    
    # 6. 根据国家查询
    print("\n6. 英国交易商:")
    print("-" * 40)
    uk_brokers = query.get_brokers_by_country('英国')
    print(uk_brokers.to_string(index=False))
    
    # 7. 根据平台查询
    print("\n7. MT4平台交易商:")
    print("-" * 40)
    mt4_brokers = query.get_brokers_by_platform('MT4')
    print(mt4_brokers.to_string(index=False))
    
    # 8. 搜索交易商
    print("\n8. 搜索包含'FX'的交易商:")
    print("-" * 40)
    search_results = query.search_brokers('FX')
    print(search_results.to_string(index=False))
    
    # 9. 根据最低存款范围查询
    print("\n9. 最低存款在$1000-$5000的交易商:")
    print("-" * 40)
    deposit_brokers = query.get_brokers_by_deposit_range(1000, 5000)
    print(deposit_brokers.to_string(index=False))
    
    # 10. 根据点差范围查询
    print("\n10. 点差在1.0-2.0 pips的交易商:")
    print("-" * 40)
    spread_brokers = query.get_brokers_by_spread_range(1.0, 2.0)
    print(spread_brokers.to_string(index=False))
    
    # 11. 具有特定特性的交易商
    print("\n11. 提供模拟账户和移动交易的交易商:")
    print("-" * 40)
    feature_brokers = query.get_brokers_with_features(['demo_account', 'mobile_trading'])
    print(feature_brokers.to_string(index=False))
    
    # 12. 获取特定交易商详细信息
    print("\n12. FXCM福汇的详细信息:")
    print("-" * 40)
    fxcm_info = query.get_comprehensive_broker_info('FXCM福汇')
    if not fxcm_info.empty:
        for column in fxcm_info.columns:
            print(f"{column}: {fxcm_info.iloc[0][column]}")
    
    query.close()

if __name__ == "__main__":
    main()