# machine-learning-handbook

按主题模块化的深度学习与机器学习及相关数学基础在线知识库，基于 MkDocs Material 构建。

## 在线访问

<https://ml-atlas-production.up.railway.app>

## 本地预览

```bash
pip install -r requirements.txt
mkdocs serve         # http://127.0.0.1:8000
mkdocs build --strict
```

## 目录

```
docs/
├── index.md
├── stylesheets/
└── T<NN>-slug/index.md
mkdocs.yml
requirements.txt
nixpacks.toml
```

新增专题：在 `docs/` 下新建 `T<NN>-slug/index.md`，并在 `mkdocs.yml` 的 `nav` 追加一行。

## License

Content licensed under [CC BY-NC-SA 4.0](./LICENSE).
