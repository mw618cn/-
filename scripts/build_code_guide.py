#!/usr/bin/env python3
"""生成可编辑Word代码使用说明书，并动态索引实际源码的函数行号。

输入为交付包中的Markdown说明和Python源码；输出为微软雅黑正文的
DOCX。这里不运行研究模型，也不重新计算结果。标题、正文、代码块和
表格均为Word原生对象；公式源保持可编辑文字，论文正式OMML公式见
独立模型Word。源文件行号由AST现场读取，避免手工维护过时索引。
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


FONT = "微软雅黑"
BLUE = "183B59"
TEAL = "127C80"
ROOT = Path(__file__).resolve().parents[1]


def set_font(run, size=None, bold=None, monospace=False):
    """同时设置西文字体和东亚字体，避免中文在Word中回退为宋体。"""
    run.font.name = "Consolas" if monospace else FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def field(paragraph, instruction):
    """插入页码等Word动态域，不把最终页数硬编码进文件。"""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    code = OxmlElement("w:instrText"); code.set(qn("xml:space"), "preserve"); code.text = instruction
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, code, end])
    set_font(run, 9)


def setup_document():
    """统一A4版式、标题层级、代码样式、页眉页脚与可编辑默认字体。"""
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.27); section.page_height = Inches(11.69)
    section.top_margin = Inches(.72); section.bottom_margin = Inches(.68)
    section.left_margin = Inches(.72); section.right_margin = Inches(.72)
    for name in ("Normal", "Body Text", "List Bullet", "List Number"):
        style = doc.styles[name]
        style.font.name = FONT; style.font.size = Pt(10)
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.line_spacing = 1.18
    for i, size in ((1, 19), (2, 14), (3, 11.5), (4, 10.5)):
        style = doc.styles[f"Heading {i}"]
        style.font.name = FONT; style.font.size = Pt(size); style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLUE if i < 3 else TEAL)
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True
    code = doc.styles.add_style("CodeBlock", 1)
    code.font.name = "Consolas"; code.font.size = Pt(8.5)
    code.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    code.paragraph_format.line_spacing = 1.05
    code.paragraph_format.space_before = Pt(2); code.paragraph_format.space_after = Pt(2)
    code.paragraph_format.left_indent = Inches(.12)
    code.paragraph_format.right_indent = Inches(.08)
    # 文档中出现的长路径/命令允许按字符换行，避免超出版心。
    ppr = code.element.get_or_add_pPr()
    wrap = OxmlElement("w:wordWrap"); wrap.set(qn("w:val"), "0"); ppr.append(wrap)
    header = section.header.paragraphs[0]
    header.text = "主—配—微协同韧性恢复｜Python代码使用说明书"
    for run in header.runs: set_font(run, 8)
    footer = section.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(footer.add_run("第 "), 9); field(footer, "PAGE")
    set_font(footer.add_run(" 页 / 共 "), 9); field(footer, "NUMPAGES")
    set_font(footer.add_run(" 页"), 9)
    doc.core_properties.title = "主配微协同韧性恢复Python代码使用说明书"
    doc.core_properties.subject = "参数、情景信息、物理模型、求解、校核、24小时出图和真实数据替换"
    doc.core_properties.author = "研究原型交付文档"
    return doc


def add_inline(paragraph, text, size=None):
    """将Markdown粗体/行内代码转为Word可编辑样式，保留正文语义。"""
    # 链接保留可读名称和目标，说明书不依赖外部Markdown渲染器。
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    tokens = re.split(r"(\*\*.*?\*\*|`[^`]+`)", text)
    for token in tokens:
        if not token:
            continue
        bold = token.startswith("**") and token.endswith("**")
        mono = token.startswith("`") and token.endswith("`")
        literal = token[2:-2] if bold else token[1:-1] if mono else token
        run = paragraph.add_run(literal)
        set_font(run, size, bold if bold else None, mono)
        if mono:
            run.font.color.rgb = RGBColor.from_string("355A69")


def shade(cell, fill):
    """设置表格背景色，用于区分表头与正文。"""
    props = cell._tc.get_or_add_tcPr()
    node = OxmlElement("w:shd"); node.set(qn("w:fill"), fill); props.append(node)


def add_table(doc, rows, widths=None):
    """把Markdown表格或动态源码索引转成真正的Word表格。"""
    if not rows:
        return
    count = max(len(row) for row in rows)
    table = doc.add_table(rows=1, cols=count)
    table.style = "Table Grid"; table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, values in enumerate(rows):
        cells = table.rows[0].cells if r == 0 else table.add_row().cells
        trpr = table.rows[r]._tr.get_or_add_trPr()
        # 一行整体换页，长行在源文件索引里依然可读。
        cant = OxmlElement("w:cantSplit"); trpr.append(cant)
        if r == 0:
            repeat = OxmlElement("w:tblHeader"); trpr.append(repeat)
        for c in range(count):
            cell = cells[c]; cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(3); p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.line_spacing = 1.08
            add_inline(p, values[c] if c < len(values) else "", 8.4 if count >= 4 else 9)
            if r == 0:
                shade(cell, "E9F0F6")
                for run in p.runs: run.bold = True
            elif r % 2 == 0:
                shade(cell, "F8FAFC")
            if widths and c < len(widths):
                cell.width = Inches(widths[c])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def markdown(doc, path):
    """解析本项目使用的基本Markdown：标题、段落、列表、代码和表格。

    说明书不把整页截图塞入Word；所有文字和表格都可点选编辑。math
    围栏明确作为LaTeX源显示；正式模型中的OMML公式由build_paper生成。
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1; continue
        if line.startswith("# "):
            # 每份源文档的标题由外层章节标题统一显示。
            i += 1; continue
        if line.startswith("```"):
            language = line[3:].strip(); code = []; i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            if language == "math":
                p = doc.add_paragraph(); set_font(p.add_run("LaTeX公式源（正式可编辑公式见模型Word）："), 9, True)
            for code_line in code:
                p = doc.add_paragraph(style="CodeBlock")
                p.paragraph_format.keep_with_next = False
                run = p.add_run(code_line if code_line else " "); set_font(run, 8.4, monospace=True)
                props = p._p.get_or_add_pPr()
                fill = OxmlElement("w:shd"); fill.set(qn("w:fill"), "F1F4F7"); props.append(fill)
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", x.replace(" ", "")) for x in cells):
                    rows.append(cells)
                i += 1
            add_table(doc, rows); continue
        heading = re.match(r"^(#{2,4})\s+(.+)$", line)
        if heading:
            p = doc.add_paragraph(style=f"Heading {min(4, len(heading.group(1)))}")
            add_inline(p, heading.group(2)); i += 1; continue
        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if bullet or numbered:
            p = doc.add_paragraph(style="List Bullet" if bullet else "List Number")
            add_inline(p, (bullet or numbered).group(1)); i += 1; continue
        # 合并一个自然段，避免Markdown换行被误处理成过多Word段落。
        parts = [line.strip()]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|```|\s*\||\s*[-*]\s|\s*\d+[.)]\s)", lines[i]):
            parts.append(lines[i].strip()); i += 1
        p = doc.add_paragraph(); add_inline(p, " ".join(parts))


class SourceIndex(ast.NodeVisitor):
    """从AST记录函数/类/方法位置，不导入并执行研究源码。"""
    def __init__(self):
        self.stack = []; self.items = []

    def visit_ClassDef(self, node):
        """类本身及类方法均列入目录，方便定位求解器装配器。"""
        self._record(node, "类"); self.stack.append(node.name)
        self.generic_visit(node); self.stack.pop()

    def visit_FunctionDef(self, node):
        """记录函数的实际起始行及docstring摘要，也包括局部工具函数。"""
        self._record(node, "函数"); self.stack.append(node.name)
        self.generic_visit(node); self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _record(self, node, kind):
        docstring = ast.get_docstring(node) or "见所属模块中文说明与分段注释"
        first = docstring.strip().splitlines()[0]
        self.items.append({"name": ".".join([*self.stack, node.name]), "kind": kind,
                           "line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno),
                           "summary": first})


def add_source_index(doc, root):
    """动态生成源文件及函数行号索引，覆盖核心包、脚本和测试文件。"""
    paths = sorted((root / "src" / "tdm_restore").glob("*.py"))
    paths += sorted((root / "scripts").glob("*.py"))
    paths += sorted((root / "tests").glob("test_*.py"))
    summary = [["源文件", "总行数", "函数/类数量", "模块用途"]]
    collected = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        index = SourceIndex(); index.visit(tree)
        module_doc = (ast.get_docstring(tree) or "包入口或测试模块").splitlines()[0]
        relative = str(path.relative_to(root))
        summary.append([relative, str(len(source.splitlines())), str(len(index.items)), module_doc])
        collected.append((relative, index.items))
    add_table(doc, summary, [2.5, .5, .6, 3.0])
    for relative, items in collected:
        if not items:
            continue
        doc.add_paragraph(relative, style="Heading 3")
        rows = [["函数/类（局部函数带所属前缀）", "源码行号", "具体作用"]]
        for item in items:
            rows.append([item["name"], str(item["line"]), item["summary"]])
        add_table(doc, rows, [2.5, .55, 3.55])
    return len(paths), sum(len(items) for _, items in collected)


def build(output):
    """按参数→场景→模型→求解→校核→图的顺序综合成可编辑说明书。"""
    doc = setup_document()
    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(72)
    set_font(title.add_run("主—配—微协同韧性恢复\nPython代码使用说明书"), 27, True)
    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(sub.add_run("24小时 · 96时段 · 参数—模型—求解—校核—出图"), 13)
    note = doc.add_paragraph(); note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(note.add_run("可编辑Word｜源码函数行号动态索引｜示例参数可替换"), 11)
    doc.add_paragraph("本说明书解释每个模块和主要代码块的职责、输入、方法、输出与适用边界。详细数学模型、OMML公式及计算算例见独立模型Word。本书中的代码和LaTeX源均保持可编辑文本。")
    doc.add_paragraph("推荐阅读顺序：快速运行 → 参数/场景 → 模型与求解 → 校核与图 → 滚动过程 → 真实数据替换 → 源码行号索引。")
    flow = [["参数与曲线", "巡检信息", "96时段MILP", "校核与D", "轨迹分解与图"],
            ["JSON / CSV", "4情景 / 8策略", "物理+因果约束", "残差 / 上下界", "条件域 / 韧性"]]
    add_table(doc, flow)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc.add_paragraph(f"构建时间：{stamp}。函数行号在构建时从实际交付源码读取；后续修改代码后请重新生成本书。")
    chapters = [
        ("第一部分  快速运行、角色与结果文件", ROOT / "README.md"),
        ("第二部分  从参数到出图的逐段代码导读", ROOT / "docs" / "code_walkthrough.md"),
        ("第三部分  真实算例参数与96点曲线替换", ROOT / "docs" / "data_replacement.md"),
        ("附录A  条件能力域、支持投影与完整轨迹分解", ROOT / "docs" / "capability.md"),
        ("附录B  结果图的读法与证据范围", ROOT / "docs" / "figures.md"),
    ]
    for heading, path in chapters:
        if not path.exists():
            raise FileNotFoundError(f"说明书源文件不存在：{path}")
        doc.add_page_break(); doc.add_paragraph(heading, style="Heading 1")
        markdown(doc, path)
    doc.add_page_break(); doc.add_paragraph("附录C  实际源文件与函数行号索引", style="Heading 1")
    doc.add_paragraph("以下表格由Python AST解析当前代码自动生成。行号为函数/类定义起始行；局部函数以所属函数作前缀。此处不粘贴整套源码，详细实现请按相对路径与行号打开代码。")
    files, definitions = add_source_index(doc, ROOT)
    doc.add_paragraph(f"本次索引覆盖{files}个Python文件、{definitions}个类/函数定义（包括局部工具函数）。")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    return output, files, definitions


def main(argv=None):
    """命令行只负责文档构建，不重新求解任何算例。"""
    parser = argparse.ArgumentParser(description="生成可编辑Word代码说明书")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "Python代码说明书_可编辑.docx")
    args = parser.parse_args(argv)
    output, files, definitions = build(args.output)
    print(f"说明书已生成：{output.resolve()}；动态索引{files}文件/{definitions}定义")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
