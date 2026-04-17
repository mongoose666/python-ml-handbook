# Converting the Handbook to PDF (Dark Nordic Theme)

This guide explains three production-ready conversion paths for:

- `handbook/python-ml-ai-handbook.md`

The Nordic palette used in all methods:

- Background: `#2e3440`
- Primary text: `#d8dee9`
- Accent 1: `#88c0d0`
- Accent 2: `#81a1c1`
- Accent 3: `#5e81ac`

---

## 1) Pandoc + LaTeX (recommended for book-quality pagination)

### Prerequisites

- Pandoc
- A LaTeX engine (`xelatex` preferred)

### Command

```bash
pandoc handbook/python-ml-ai-handbook.md \
  --defaults handbook/styles/pandoc-nordic.yaml \
  -o handbook/python-ml-ai-handbook-pandoc.pdf
```

### Notes

- The provided defaults file applies a dark Nordic page, text, and link colors.
- You can tune geometry and fonts in `handbook/styles/pandoc-nordic.yaml`.

---

## 2) Typst

Typst is fast and clean for modern technical PDF output.

### Option A: Convert Markdown directly with Pandoc to Typst, then compile

```bash
pandoc handbook/python-ml-ai-handbook.md -t typst -o /tmp/handbook.typ
```

Then add a small Typst wrapper (theme colors, page style), and compile:

```bash
typst compile /tmp/handbook.typ handbook/python-ml-ai-handbook-typst.pdf
```

### Option B: Maintain a custom `.typ` template

If you prefer full control, keep a dedicated Typst template and include the markdown content there.

---

## 3) WeasyPrint (HTML/CSS pipeline)

### Prerequisites

- WeasyPrint
- A markdown-to-HTML converter (Pandoc or Python-Markdown)

### Step 1: Markdown → HTML

```bash
pandoc handbook/python-ml-ai-handbook.md -s -o /tmp/handbook.html
```

### Step 2: HTML + CSS → PDF

```bash
weasyprint /tmp/handbook.html handbook/python-ml-ai-handbook-weasyprint.pdf \
  -s handbook/styles/nordic-theme.css
```

### Sample CSS usage

The included CSS file is:

- `handbook/styles/nordic-theme.css`

It provides:
- Nord color palette
- Cover page styling (first H1 as cover)
- TOC styling
- Code block styling
- Print-friendly margins and page-break behavior

---

## Optional: quick custom CSS snippet

```css
body {
  background: #2e3440;
  color: #d8dee9;
}
a { color: #88c0d0; }
```

For full styling, use the provided stylesheet: `handbook/styles/nordic-theme.css`.
