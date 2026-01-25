# Documentation Setup Guides

This directory contains **reusable guides** for setting up documentation infrastructure across projects.

---

## Contents

### 1. [mkdocs_mathjax_setup_guide.md](mkdocs_mathjax_setup_guide.md)

**Comprehensive guide for MkDocs + MathJax setup**

- Complete step-by-step instructions
- Configuration file templates
- Troubleshooting section
- Examples and best practices
- ~20 pages, covers everything

**Use when:** Setting up documentation for a new project from scratch

---

### 2. [mkdocs_quick_reference.md](mkdocs_quick_reference.md)

**Quick reference and cheat sheet**

- Essential commands
- Math syntax reference
- Common patterns
- Troubleshooting quick fixes
- 5-minute setup guide

**Use when:** You need a quick reminder or reference while writing docs

---

## Why These Guides?

### Problem

GitHub's markdown rendering struggles with LaTeX math:

```
The equation x_u = (Y C_u Y^T + λI)^{-1} Y C_u r_u
```

Shows as **raw text** ❌ - completely unreadable for research/ML projects!

### Solution

**MkDocs + MathJax** renders equations beautifully:

$$
x_u = (Y C_u Y^T + \lambda I)^{-1} Y C_u r_u
$$

Professional, publication-quality math rendering ✅

---

## Quick Start

### For a New Project

1. **Copy configuration files** from [mkdocs_mathjax_setup_guide.md](mkdocs_mathjax_setup_guide.md)
2. **Customize** site name, URLs, navigation
3. **Test locally:** `mkdocs serve`
4. **Deploy:** `mkdocs gh-deploy`
5. **Configure GitHub Pages** (one-time)

**Time:** 15-20 minutes

---

### For CF-Ensemble (This Project)

MkDocs + MathJax is **already set up!**

```bash
# Activate environment
mamba activate cfensemble

# Write documentation
vim docs/methods/your-topic.md

# Test locally with live reload
mkdocs serve

# Commit and push (auto-deploys)
git add docs/methods/your-topic.md
git commit -m "Add documentation"
git push origin main
```

**Site:** https://pleiadian53.github.io/cf-ensemble/

---

## Key Features

✅ **LaTeX Math Rendering**
- Inline: `$E = mc^2$`
- Display: `$$\int_0^\infty e^{-x} dx$$`
- Multi-line equations with `align`

✅ **Professional Theme**
- Material theme with dark mode
- Mobile-responsive
- Full-text search

✅ **Auto-Deployment**
- Push to main → site updates automatically
- GitHub Actions workflow
- 2-3 minutes deployment time

✅ **Jupyter Notebook Support**
- Render `.ipynb` files directly
- Include outputs and plots

✅ **Mermaid Diagrams**
- Flowcharts, sequence diagrams
- Rendered as SVG

---

## When to Use

### Use MkDocs + MathJax when:

- ✅ Your project has **equations/math notation**
- ✅ You want **professional documentation**
- ✅ You need **better than GitHub rendering**
- ✅ You want **auto-deployment**
- ✅ You need **search functionality**

### Don't use when:

- ❌ Simple project with no math
- ❌ Just a README is sufficient
- ❌ No time for setup

---

## Documentation Structure

**Recommended organization:**

```
docs/
├── index.md                    # Landing page
├── getting-started/            # For new users
│   ├── installation.md
│   └── quickstart.md
├── tutorials/                  # Step-by-step guides
├── how-to/                     # Problem-specific
├── reference/                  # API docs
├── theory/                     # Math/algorithms
└── development/                # For contributors
```

**Principle:** Organize by **user intent**, not code structure.

---

## Best Practices

### 1. Math Notation

```markdown
# Define notation upfront
## Notation
- $m$ : number of classifiers
- $n$ : number of instances
- $k$ : latent dimension

# Use inline for simple
The quality $q$ is defined as...

# Use display for complex
$$
\mathcal{L} = \rho \|R - XY^T\|_F^2 + (1-\rho) \sum_{i} (r_i - y_i)^2
$$
```

---

### 2. Code Examples

```python
# Complete and runnable
from mypackage import Model

# Create model
model = Model(param=10)

# Fit and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Expected output
print(predictions.shape)  # (1000,)
```

---

### 3. Progressive Complexity

Start simple → intermediate → advanced

```markdown
# Basic Tutorial
Basic usage for beginners

# Advanced Tutorial
Complex scenarios and edge cases

# Theory and Math
Deep dive into algorithms
```

---

### 4. Testing

```bash
# Always test before pushing
mkdocs serve              # Visual check
mkdocs build --strict     # Catch broken links
```

---

## Common Issues

### Math not rendering?

1. Check `pymdownx.arithmatex: generic: true` in `mkdocs.yml`
2. Verify `mathjax.js` exists in `docs/javascripts/`
3. Clear browser cache

### Build fails?

1. Validate YAML: `python -c "import yaml; yaml.safe_load(open('mkdocs.yml'))"`
2. Check for broken links: `mkdocs build --strict`
3. Ensure all files in `nav` exist

### Site not updating?

1. Check GitHub Actions: `github.com/user/repo/actions`
2. Verify workflow ran successfully
3. Hard refresh browser: Ctrl+Shift+R

**See [mkdocs_mathjax_setup_guide.md](mkdocs_mathjax_setup_guide.md) for detailed troubleshooting.**

---

## Resources

### Official Documentation

- **MkDocs:** https://www.mkdocs.org/
- **Material Theme:** https://squidfunk.github.io/mkdocs-material/
- **MathJax:** https://docs.mathjax.org/

### Tutorials

- **Math Syntax:** https://math.meta.stackexchange.com/questions/5020/
- **Mermaid Diagrams:** https://mermaid.live/

### Examples

Real projects using MkDocs:
- **FastAPI:** https://fastapi.tiangolo.com/
- **Pydantic:** https://docs.pydantic.dev/
- **Ray:** https://docs.ray.io/

---

## Maintenance

### Updating These Guides

When updating setup procedures:

1. **Test on a fresh project** to ensure instructions work
2. **Update both guides** (full + quick reference)
3. **Note date** in "Last Updated" section
4. **Commit changes** with clear message

### Version Compatibility

**Tested with:**
- MkDocs 1.6.1
- mkdocs-material 9.7.1
- Python 3.10+
- pymdown-extensions 10.20.1

---

## History

### 2026-01-24

- Initial creation
- Comprehensive setup guide (mkdocs_mathjax_setup_guide.md)
- Quick reference (mkdocs_quick_reference.md)
- Successfully applied to cf-ensemble project
- Site live at: https://pleiadian53.github.io/cf-ensemble/

---

## Contributing

Found an issue or improvement?

1. Test your fix on a real project
2. Update relevant guide(s)
3. Update version/date information
4. Document what changed

---

## Summary

**These guides enable professional documentation with LaTeX math rendering for any Python/ML/research project.**

- **Setup time:** 15-20 minutes (one-time)
- **Maintenance:** Automatic via GitHub Actions
- **Result:** Beautiful, searchable documentation site

**Your research deserves professional documentation!** 📚✨

---

**Location:** `docs/guides/`  
**Status:** Production-ready, tested on cf-ensemble  
**Last Updated:** 2026-01-24
