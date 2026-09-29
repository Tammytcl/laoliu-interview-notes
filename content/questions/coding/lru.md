---
id: lru-cache
title: "如何实现 O(1) 的 LRU Cache？"
category: coding
difficulty: 进阶
tags: ["数据结构","哈希表"]
updated: 2026-09-30
summary: "使用哈希表定位节点，使用双向链表维护最近使用顺序。"
draft: false
---

## 一句话回答

用哈希表将 key 映射到双向链表节点。访问时将节点移到表头；超过容量时移除表尾节点。

## 核心思路

下面是使用 Python 标准库表达思路的版本；如果面试要求手写，应实现双向链表及节点移动。

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
```

## 面试追问

- 为什么单向链表不方便做 O(1) 删除？
- 并发访问需要考虑什么？
- LRU 与 LFU 的策略区别是什么？

## 常见误区

哈希表操作的 O(1) 通常是平均复杂度。不要遗漏容量为 0、更新已有 key 和重复访问的测试。
