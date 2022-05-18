---
title:  "Awesome tools that I use"
last_modified_at: 2020-05-09
categories:
  - Blog
tags:
  - Tools
---

This is an incomplete list of awesome software tools that I use for work and for personal use.
I may write articles describing the tools and how I use them, but this page will contain brief summaries.

* TOC
{:toc}



## Text editing - LunarVim
I'm currently using [Neovim](https://neovim.io/) with [LunarVim](https://github.com/LunarVim/LunarVim) on top. 
There's no good reason to stick to vanilla Vim these days in my opinion.
NeoVim offers some exciting tools and capabilities that aren't available in vanilla Vim.

LunarVim is a pseudo-IDE layer built on top of NeoVim.
I say 'pseudo-IDE' because chances are that it will never be able to compete with a full modern IDE experience.
But in the mean time, it offers much faster startup time than typical IDEs and a pleasant vim-like text-editing experience.

I've used [Spacemacs](https://www.spacemacs.org/) for a long time in the past but I found emacs-based distributions too slow for my liking on my old laptop.
There are also some alternatives in the vim-world, like [SpaceVim](https://spacevim.org/), [doom-nvim](https://github.com/NTBBloodbath/doom-nvim/), and [NvChad](https://github.com/NvChad/NvChad).
I found LunarVim to be closest to what I want out of the box, with some minimal IDE features and a [which-key](https://github.com/justbur/emacs-which-key) like command popups so you don't have to memorize keyboard bindings.

## VSpaceCode
[VSpaceCode](https://vspacecode.github.io/) is an awesome extension for VS Code that creates Spacemacs-like keyboard bindings.
It makes using VS Code extremely fun.

## Jupytext
Jupyter notebooks are very popular these days, but there are some big pain points when using them for your code.
For example, `.ipynb` notebooks produce huge git diffs even for small insignificant changes.

[Jupytext](https://jupytext.readthedocs.io/en/latest/index.html) lets you edit notebooks as plain text documents, so you can edit them in your IDE/editor of choice.
The `.py` files can be synchronized to `.ipynb` notebooks, so you can see the changes reflected in your Jupyter interface.

## Pandoc
[Pandoc](https://pandoc.org/) is a tool that lets you convert simple documents between markup formats, e.g. markdown, html, LaTeX.
I'm a big believer in keeping personal notes in plain text documents (markdown in my case) for better searchability.
I use Pandoc to transform these notes into various useful formats, including PDF, Word and PowerPoint documents.