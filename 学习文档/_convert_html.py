# -*- coding: utf-8 -*-
"""飞书导出 HTML -> Markdown 转换脚本 v2（显式 UTF-8）

针对飞书导出特性：
- 每个有序列表项是独立的 <ol start="N">，需聚合连续同类列表并用 start 编号
- 顶层行内标签（a/b/code 等）不能丢行内语义
- 块间统一空行
- 图片 data URI 截断为占位符
"""
import sys
import re
sys.stdout.reconfigure(encoding="utf-8")

from bs4 import BeautifulSoup, NavigableString, Tag


def escape_text(text: str) -> str:
    """普通文本中的 markdown 特殊字符转义（代码块/行内代码除外）"""
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`")
    text = re.sub(r"([*_])", r"\\\1", text)
    return text


def img_placeholder(src: str) -> str:
    if not src:
        return "![](image)"
    if src.startswith("data:"):
        return "![](" + src.split(",")[0] + "...)"
    return "![](" + src + ")"


def convert_inline(node) -> str:
    """递归转换行内节点"""
    if isinstance(node, NavigableString):
        return escape_text(str(node))
    if not isinstance(node, Tag):
        return ""
    name = node.name
    if name in ("b", "strong"):
        return wrap_emphasis("".join(convert_inline(c) for c in node.children), "**")
    if name in ("i", "em"):
        return wrap_emphasis("".join(convert_inline(c) for c in node.children), "*")
    if name == "code":
        return "`" + node.get_text() + "`"
    if name == "a":
        inner = "".join(convert_inline(c) for c in node.children).strip()
        href = node.get("href", "")
        return f"[{inner}]({href})" if href else inner
    if name == "br":
        return "\n"
    if name == "img":
        return img_placeholder(node.get("src", ""))
    # span 及未知行内标签：透明递归
    return "".join(convert_inline(c) for c in node.children)


def wrap_emphasis(inner: str, mark: str) -> str:
    stripped = inner.strip()
    if not stripped:
        return inner
    lead = inner[: len(inner) - len(inner.lstrip())]
    trail = inner[len(inner.rstrip()):]
    return f"{lead}{mark}{stripped}{mark}{trail}"


def code_block_from_pre(pre: Tag) -> str:
    code = pre.find("code")
    lang = ""
    if code:
        for c in code.get("class") or []:
            if c.startswith("language-"):
                lang = c[len("language-"):]
                break
    text = (code.get_text() if code else pre.get_text()).strip("\n")
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{lang}\n{text}\n{fence}"


# ---------------- 列表（支持飞书"每项一个 ol"的聚合） ----------------

def group_consecutive_lists(children):
    """把 children 序列切分为 [块节点] 与 [连续同类列表组] 的混合流"""
    items = []  # ("node", node) 或 ("listgroup", [tags])
    kids = [c for c in children
            if not (isinstance(c, NavigableString) and not str(c).strip())]
    i = 0
    while i < len(kids):
        node = kids[i]
        if isinstance(node, Tag) and node.name in ("ul", "ol"):
            group, j = [node], i + 1
            while (j < len(kids) and isinstance(kids[j], Tag)
                   and kids[j].name == node.name):
                group.append(kids[j])
                j += 1
            items.append(("listgroup", group))
            i = j
        else:
            items.append(("node", node))
            i += 1
    return items


def convert_list_group(group, ordered: bool, indent: int = 0) -> str:
    """一组连续同类列表 -> markdown。编号优先取 ol 的 start 属性"""
    lines = []
    idx = 1
    for tag in group:
        # ol 的 start 属性决定起始编号
        if ordered and tag.get("start"):
            try:
                idx = int(tag.get("start"))
            except (TypeError, ValueError):
                pass
        for li in tag.find_all("li", recursive=False):
            marker = f"{idx}. " if ordered else "* "
            prefix = " " * indent + marker
            # li children：行内部分 + 嵌套列表组
            inline_parts = []
            for kind, item in group_consecutive_lists(li.children):
                if kind == "node":
                    n = item
                    if isinstance(n, Tag) and n.name == "p":
                        inline_parts.append(
                            "".join(convert_inline(c) for c in n.children))
                    else:
                        inline_parts.append(convert_inline(n))
                else:
                    # 嵌套列表组（紧跟在行内文本后，另起一行）
                    sub_ordered = item[0].name == "ol"
                    sub = convert_list_group(item, sub_ordered, indent + 3)
                    inline_parts.append("\n" + sub.rstrip("\n"))
            text = "".join(inline_parts).strip()
            text = re.sub(r"\n{2,}", "\n", text)
            first = True
            for line in text.split("\n"):
                if first:
                    lines.append(prefix + line)
                    first = False
                else:
                    lines.append(" " * (indent + len(marker)) + line)
            idx += 1
    return "\n".join(lines) + "\n"


# ---------------- 块级 ----------------

def convert_table(table: Tag) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""
    md_rows = []
    for tr in rows:
        cells = tr.find_all(["th", "td"])
        row = []
        for cell in cells:
            text = "".join(convert_inline(c) for c in cell.children)
            text = text.strip().replace("\n", " ").replace("|", "\\|")
            row.append(text)
        md_rows.append("| " + " | ".join(row) + " |")
    header = md_rows[0]
    ncol = max(header.count("|") - 1, 1)
    sep = "| " + " | ".join(["---"] * ncol) + " |"
    return "\n".join([header, sep] + md_rows[1:])


def process_children(children, out: list) -> None:
    """处理子节点流：聚合连续列表，其余走 convert_block"""
    for kind, item in group_consecutive_lists(children):
        if kind == "listgroup":
            ordered = item[0].name == "ol"
            out.append(convert_list_group(item, ordered).rstrip("\n") + "\n\n")
        else:
            node = item
            if isinstance(node, NavigableString):
                text = str(node).strip()
                if text:
                    out.append(escape_text(text) + "\n\n")
                continue
            convert_block(node, out)


def convert_block(node: Tag, out: list) -> None:
    name = node.name
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        text = "".join(convert_inline(c) for c in node.children).strip()
        out.append("#" * level + " " + text + "\n\n")
    elif name == "p":
        text = "".join(convert_inline(c) for c in node.children).strip()
        if text:
            out.append(text + "\n\n")
    elif name == "pre":
        out.append(code_block_from_pre(node) + "\n\n")
    elif name == "blockquote":
        inner = []
        process_children(node.children, inner)
        text = "".join(inner).rstrip("\n")
        quoted = "\n".join("> " + ln if ln.strip() else ">"
                           for ln in text.split("\n"))
        out.append(quoted + "\n\n")
    elif name == "table":
        out.append(convert_table(node) + "\n\n")
    elif name == "hr":
        out.append("---\n\n")
    elif name in ("div", "section", "article", "body"):
        process_children(node.children, out)
    else:
        # 顶层行内标签（如 a、span）：保留行内语义
        text = convert_inline(node).strip()
        if text:
            out.append(text + "\n\n")


def convert_file(src: str, dst: str) -> None:
    with open(src, "rb") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    out = []
    process_children(soup.body.children, out)
    md = "".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    # 合并开头"原文链接："与下一行标题链接（吞掉中间的 > 空行）
    md = re.sub(r"(原文链接：)\s*(?:\n>?)+\s*", r"\1", md, count=1)
    md = md.strip() + "\n"

    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(f"OK: {dst}")
    print(f"  chars: {len(md)}")
    print("---- preview ----")
    print(md[:400])
    # 统计图片占位符
    print(f"  images: {md.count('![](data:')}")


if __name__ == "__main__":
    jobs = [
        (r"C:\Users\xxy\Downloads\第3课-vLLM+核心组件-WorkerExecutor+详解及协作.html",
         r"F:\Projects\vllm\学习文档\第3课-vLLM+核心组件-WorkerExecutor+详解及协作.md"),
        (r"C:\Users\xxy\Downloads\第6课-vLLM+核心组件-ModelRunner之模型加载详解.html",
         r"F:\Projects\vllm\学习文档\第6课-vLLM+核心组件-ModelRunner之模型加载详解.md"),
    ]
    for src, dst in jobs:
        convert_file(src, dst)
        print("=" * 60)
