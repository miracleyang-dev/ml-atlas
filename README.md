# ink-ml-notes

Handwritten notes on ML/DL & math fundamentals, organized by topic modules for high-frequency review.

> 手写整理的机器学习 / 深度学习核心知识点与数学基础速查库。按 **专题模块化** 划分，不严格绑定学科边界。

---

## Topics 专题索引

| ID  | Topic                          | Keywords                          | Status |
| --- | ------------------------------ | --------------------------------- | ------ |
| T01 | SVD / PCA / LDA                | dim reduction · eigen · projection | 🚧 WIP |
| T02 | Matrix Calculus 矩阵求导       | layout · chain rule · Jacobian    | 🚧 WIP |
| T03 | KKT & Duality 对偶问题         | convex · Lagrangian · slackness   | 🚧 WIP |
| T04 | DL Optimizers 优化器           | SGD · Momentum · Adam · Lion      | 🚧 WIP |

Full list lives in [`INDEX.md`](./INDEX.md).

## Layout 目录约定

```
topics/
└── TNN-slug/
    ├── README.md   # 1-3 sentences + keywords + refs
    ├── 01.png      # zero-padded, append-only
    └── 02.png
```

- **PNG-first**: native images render inline on GitHub, single-page edits are friction-free.
- **Append-only IDs**: 专题编号只增不改，旧链接永不失效。
- **PDF optional**: 某专题封板后，可在该目录追加 `topic.pdf` 作为离线副本。

## Conventions 编辑约定

- File naming: `NN.png` (zero-padded two digits)
- New topic = new `TNN-` folder, never renumber existing ones
- Large PNGs tracked via **Git LFS** (see `.gitattributes`)
- Commit messages: `T03: add KKT slackness page` / `T01: refine SVD geometry diagram`

## License

Content licensed under [CC BY-NC-SA 4.0](./LICENSE) — non-commercial use with attribution, share-alike.
