# Developer Guide

## Environment Setup

Besides the environment setup in [Environment Setup](env-setup.md), you need to install `mkdocs-material` for maintaining the documentation.

```bash
pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-git-committers-plugin-2
```

## Documentation

Currently we maintain the **deliverable documentation** in `docs` folder. The documentation is built using [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), and each markdown file is a page in the documentation. Everytime a new commit is pushed to the `main` branch, the documentation will be automatically built and deployed to [GitHub Pages](https://aicrosssim.github.io/NewComputeBench/).

### How to add a new page?

1. Create a new markdown file in the `docs` folder.
2. Add the new page to the `mkdocs.yml` file under the `nav` section.

### How to preview the documentation?

Run the following command in the root directory of the project:

```bash
mkdocs serve
```

Then you can preview the static sites in your browser.

## How to add a new transform-aware pretraining?