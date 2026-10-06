#!/bin/sh
# Reproduce everything: models -> tables -> figures -> paper PDF
set -e
Rscript wda_sthnet.R
Rscript make_tables_tex.R
mkdir -p paper/figures paper/tables
cp output/figures/*.png paper/figures/; cp output/tables/*.tex paper/tables/; cp wda_sthnet.R make_tables_tex.R paper/
cd paper && pdflatex -interaction=nonstopmode wda_sthnet_paper.tex && pdflatex -interaction=nonstopmode wda_sthnet_paper.tex && pdflatex -interaction=nonstopmode wda_sthnet_paper.tex
