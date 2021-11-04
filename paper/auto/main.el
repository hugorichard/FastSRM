(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("algorithm2e" "ruled" "vlined")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "caption"
    "wrapfig"
    "inputenc"
    "comment"
    "fontenc"
    "hyperref"
    "xurl"
    "booktabs"
    "amsfonts"
    "nicefrac"
    "microtype"
    "todonotes"
    "longtable"
    "algorithm2e"
    "amsmath"
    "utils/kky"
    "subcaption"
    "afterpage"
    "float"
    ""
    "bbold")
   (LaTeX-add-labels
    "sec:srm:review"
    "eq:model:srm"
    "fig:srm:conceptual_figure"
    "sec:deterministicsrm"
    "eq:detsrmloss"
    "eq:srm:supdate"
    "eq:detsrm:Aiupdate"
    "sec:probabilisticsrm"
    "probsrmcompleted"
    "eq:psrm:Aiupdate"
    "fig:srm:conceptual"
    "sec:optimal_atlas"
    "prop:optimaldetsrm"
    "eq:fastdetsrm:Aiupdate"
    "eq:equality:xy"
    "prop:optimalprobsrm"
    "eq:fastprobsrm:Aiupdate"
    "fig:srm:synthetic_gradient"
    "exp:identifiability"
    "sec:timesegment_expe"
    "timesegment_expe"
    "fig:srm:timesegment")
   (LaTeX-add-bibliographies
    "utils/biblio"))
 :latex)

