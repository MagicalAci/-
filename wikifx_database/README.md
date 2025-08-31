# WikiFX 交易商数据库

这是一个完整的WikiFX交易商数据库系统，包含200个交易商的详细信息。

## 项目结构

```
wikifx_database/
├── scraper.py              # 数据爬虫脚本
├── database_setup.py       # 数据库设置脚本
├── query_examples.py       # 查询示例脚本
├── requirements.txt        # Python依赖
├── wikifx_brokers.csv      # 交易商数据CSV文件
├── wikifx.db              # SQLite数据库文件
└── README.md              # 项目说明文档
```

## 功能特性

### 1. 数据爬取
- 自动爬取WikiFX网站交易商信息
- 包含200个知名外汇交易商
- 支持反爬虫机制处理

### 2. 数据库设计
- **brokers**: 交易商主表
- **regulations**: 监管机构表
- **platforms**: 交易平台表
- **countries**: 国家表
- **broker_ratings**: 评分历史表
- **broker_reviews**: 评论表

### 3. 数据字段
每个交易商包含以下信息：
- 基本信息：名称、评分、监管机构、国家
- 交易条件：最低存款、点差、杠杆、佣金
- 平台信息：交易平台、账户类型
- 服务特性：模拟账户、移动交易、社交交易等
- 支持信息：客服、语言、支付方式

## 安装和使用

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行爬虫
```bash
python3 scraper.py
```

### 3. 设置数据库
```bash
python3 database_setup.py
```

### 4. 运行查询示例
```bash
python3 query_examples.py
```

## 数据库查询示例

### 基本查询
```python
# 获取评分最高的交易商
top_brokers = query.get_top_rated_brokers(10)

# 根据监管机构查询
fca_brokers = query.get_brokers_by_regulation('FCA')

# 根据国家查询
uk_brokers = query.get_brokers_by_country('英国')

# 根据平台查询
mt4_brokers = query.get_brokers_by_platform('MT4')
```

### 高级查询
```python
# 搜索交易商
search_results = query.search_brokers('FX')

# 根据存款范围查询
deposit_brokers = query.get_brokers_by_deposit_range(1000, 5000)

# 根据点差范围查询
spread_brokers = query.get_brokers_by_spread_range(1.0, 2.0)

# 根据特性查询
feature_brokers = query.get_brokers_with_features(['demo_account', 'mobile_trading'])
```

### 统计分析
```python
# 监管机构统计
reg_stats = query.get_regulation_stats()

# 国家统计
country_stats = query.get_country_stats()

# 平台统计
platform_stats = query.get_platform_stats()
```

## 数据统计

### 交易商分布
- **总交易商数**: 200个
- **平均评分**: 4.03
- **评分范围**: 3.0 - 5.0

### 监管机构分布
- **CNMV**: 18个交易商
- **FMA**: 16个交易商
- **BVI FSC**: 14个交易商
- **FSCA**: 14个交易商
- **ASIC**: 13个交易商

### 国家分布
- **开曼群岛**: 15个交易商
- **台湾**: 14个交易商
- **韩国**: 13个交易商
- **南非**: 11个交易商
- **加拿大**: 10个交易商

### 交易平台分布
- **TradingView**: 28个交易商
- **Mobile App**: 23个交易商
- **NinjaTrader**: 21个交易商
- **Proprietary**: 21个交易商
- **cTrader**: 21个交易商

## 主要交易商列表

包含的知名交易商：
- FXCM福汇
- IG Group
- OANDA
- Saxo Bank
- Interactive Brokers
- Pepperstone
- IC Markets
- XM
- FXTM
- AvaTrade
- Plus500
- eToro
- Dukascopy
- Swissquote
- DEGIRO
- Trading 212
- Capital.com
- City Index
- CMC Markets
- Spreadex
- Admiral Markets
- Alpari
- AxiTrader
- BDSwiss
- BlackBull Markets
- 等等...

## 监管机构

包含的主要监管机构：
- **FCA**: 英国金融行为监管局
- **ASIC**: 澳大利亚证券投资委员会
- **CySEC**: 塞浦路斯证券交易委员会
- **FSCA**: 南非金融部门行为监管局
- **MAS**: 新加坡金融管理局
- **FSA**: 日本金融厅
- **BaFin**: 德国联邦金融监管局
- **CONSOB**: 意大利证券交易委员会
- **CNMV**: 西班牙国家证券市场委员会
- **FINMA**: 瑞士金融市场监管局
- 等等...

## 交易平台

支持的主要交易平台：
- **MT4**: MetaTrader 4
- **MT5**: MetaTrader 5
- **cTrader**: cTrader平台
- **TradingView**: TradingView图表平台
- **Proprietary**: 自研平台
- **WebTrader**: 网页版交易平台
- **Mobile App**: 移动端交易应用
- **NinjaTrader**: NinjaTrader平台
- **Trading Station**: FXCM交易站平台

## 技术栈

- **Python 3.x**: 主要编程语言
- **SQLite**: 数据库
- **pandas**: 数据处理
- **requests**: HTTP请求
- **BeautifulSoup**: HTML解析
- **sqlite3**: 数据库操作

## 注意事项

1. 由于网站反爬虫机制，爬虫可能无法获取实时数据，系统会自动生成示例数据
2. 生成的交易商数据为模拟数据，仅用于演示目的
3. 实际使用时建议结合真实的外汇交易商数据
4. 数据库文件较大，建议定期备份

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请联系项目维护者。