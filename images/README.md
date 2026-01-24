# Visual Assets for CF-Ensemble

This directory contains visual assets (diagrams, illustrations, logos) used throughout the project documentation.

## Current Assets

### Diagrams Referenced from GitHub
The main README currently references diagrams hosted on GitHub:

1. **Workflow Diagram**: https://user-images.githubusercontent.com/1761957/188764919-f2217d9f-c451-4c51-9b34-cde9f8cdc7b4.png
   - Shows the basic CF-Ensemble workflow
   - Used in: `README.md` (line ~119)

2. **Optimization Formula**: https://user-images.githubusercontent.com/1761957/188937553-e74e9837-51cf-4c7e-8ef9-66146ceb8d95.png
   - Mathematical formulation of the objective
   - Used in: `README.md` (line ~155)

### Mermaid Diagrams
The README uses Mermaid for inline diagrams (rendered by GitHub):
- Workflow flowchart (README.md, line ~112)
- Future: Can add more interactive diagrams

## Creating New Visual Assets

### Tools Recommended

1. **Diagrams & Flowcharts**
   - [Mermaid](https://mermaid.js.org/) - Text-based diagrams (renders in GitHub)
   - [Draw.io](https://draw.io/) - Visual diagram editor
   - [Excalidraw](https://excalidraw.com/) - Hand-drawn style diagrams

2. **Illustrations**
   - [Inkscape](https://inkscape.org/) - Vector graphics (SVG)
   - [GIMP](https://www.gimp.org/) - Raster graphics (PNG)

3. **Mathematical Diagrams**
   - LaTeX with TikZ
   - [LaTeXiT](https://www.chachatelier.fr/latexit/) (macOS)
   - [Mathcha](https://www.mathcha.io/) - Online equation editor

### Style Guide

- **Format**: PNG or SVG (prefer SVG for scalability)
- **Resolution**: 300 DPI minimum for PNG
- **Size**: Optimize file sizes (use tools like `pngquant`, `svgo`)
- **Colors**: 
  - Primary: `#4A90E2` (blue)
  - Success: `#90EE90` (light green)
  - Accent: `#FFD700` (gold)
  - Background: White or transparent
- **Fonts**: System fonts or embedded fonts in SVG

### Naming Convention

```
{purpose}_{type}_{variant}.{ext}

Examples:
- workflow_diagram_v1.png
- optimization_formula_annotated.svg
- architecture_overview_simple.png
- confidence_comparison_chart.png
```

## Future Assets to Create

### High Priority

1. **Architecture Diagram**
   - Show package structure visually
   - Connections between modules
   - Data flow through the system

2. **Performance Visualization**
   - ROC curves comparing strategies
   - Bar charts of improvements
   - Training time comparisons

3. **Concept Illustrations**
   - CF matrix factorization visualization
   - Ensemble transformation concept
   - Reliability weight learning process

### Medium Priority

4. **Logo**
   - Project logo/icon
   - Favicon for documentation site
   - Social media preview image

5. **Tutorial Diagrams**
   - Visual guides for each tutorial
   - Step-by-step process diagrams
   - Annotated code flow

6. **Results Showcase**
   - Example outputs from experiments
   - Before/after comparisons
   - Success stories

## Usage in Documentation

### Markdown Syntax

```markdown
# Local image
![Alt text](../images/workflow_diagram.png)

# With size control (HTML)
<img src="../images/workflow_diagram.png" width="600" alt="Workflow">

# Centered with alignment
<div align="center">
<img src="../images/workflow_diagram.png" width="500">
</div>

# External image (GitHub hosted)
![Workflow](https://user-images.githubusercontent.com/1761957/188764919-f2217d9f-c451-4c51-9b34-cde9f8cdc7b4.png)
```

### Mermaid Diagrams (Inline)

````markdown
```mermaid
graph LR
    A[Start] --> B[Process]
    B --> C[End]
```
````

## Contributing Visual Assets

When adding new visual assets:

1. **Create** the asset following the style guide
2. **Optimize** file size while maintaining quality
3. **Place** in appropriate subdirectory:
   ```
   images/
   ├── diagrams/
   ├── screenshots/
   ├── charts/
   ├── logos/
   └── tutorials/
   ```
4. **Document** in this README
5. **Reference** in relevant documentation files

## Tools for Diagram Generation

### From Code

Create diagrams programmatically:

```python
# Using matplotlib for charts
import matplotlib.pyplot as plt

# Using graphviz for graphs
from graphviz import Digraph

# Using plotly for interactive plots
import plotly.graph_objects as go
```

### Export from Jupyter Notebooks

Many visualizations in `examples/` and `notebooks/` can be exported:

```bash
# Save figure in notebook
plt.savefig('../images/charts/my_plot.png', dpi=300, bbox_inches='tight')

# Or export entire notebook as PDF with figures
jupyter nbconvert --to pdf notebook.ipynb
```

## Maintenance

- 🗓️ **Review quarterly**: Check for outdated diagrams
- 🔄 **Update**: When features change, update visuals
- 🧹 **Cleanup**: Remove unused assets
- 📦 **Optimize**: Compress images to reduce repo size

## License

Visual assets in this directory follow the same MIT License as the project unless otherwise noted.

Original diagrams and illustrations created for this project may be reused with attribution.
