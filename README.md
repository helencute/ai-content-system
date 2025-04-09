# ai-content-system
ai-content-system/
├── core/
│   ├── agents/
│   │   ├── research_planner.py       # 研究计划生成
│   │   ├── source_collector.py       # 资源爬取与可信度评估
│   │   └── data_analyzer.py          # 数据分析与可视化
├── content_engine/
│   ├── generators/
│   │   ├── blog_generator.py         # 博客内容生成
│   │   ├── thread_generator.py       # 推文内容生成
│   │   └── report_generator.py       # 研究报告生成
│   └── style_transfer/               # 风格迁移模块
│       ├── brand_profiles/           # 品牌风格配置
│       └── style_adapter.py          # 风格适配器
├── integrations/
│   ├── mcp_client/                   # MCP协议集成
│   │   ├── connectors/
│   │   │   ├── wordpress_client.py   # WordPress发布
│   │   │   └── threads_client.py     # Threads发布
│   │   └── mcp_bus.py                # MCP消息总线
│   └── apis/
│       ├── news_api.py               # 新闻源接口
│       └── scholar_api.py            # 学术资源接口
├── infrastructure/
│   ├── config_loader.py              # 统一配置管理
│   ├── dependency_injection.py       # 依赖注入容器
│   └── logging/                      # 日志监控系统
├── tests/
│   ├── unit/
│   │   ├── test_research_planner.py
│   │   └── test_style_adapter.py
│   └── integration/
│       └── test_pipeline_e2e.py
├── docs/
│   ├── api_reference.md              # 模块接口文档
│   └── deployment_guide.md           # 部署指南
├── scripts/
│   ├── deploy.sh                     # 部署脚本
│   └── setup_mcp.sh                  # MCP环境配置
├── config/
│   ├── mcp_config.yaml               # MCP协议配置
│   └── brand_profiles/               # 品牌风格配置文件
│       ├── tech_blog.yaml
│       └── social_media.yaml
├── Dockerfile                        # 容器化配置
├── requirements.txt                  # Python依赖
└── main.py                           # 系统入口
