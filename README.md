# V8 SQL Augmentation

Spider数据集SQL数据增强流水线。基于LLM合成+共识投票+反向编译，生成两个复杂度层级的查询（moderate_low [3-4], moderate_high [5-7]）。

运行: `python run.py`，需要在 `.env` 中配置 `DASHSCOPE_API_KEY`。
