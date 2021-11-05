NAMES = {
    "probsrm": "ProbSRM",
    "detsrm": "DetSRM",
}


def add(NAMES, algo, atlas_only=False, display_regions=True):
    dic = {}
    for a in algo:
        if "fastsrm" not in a:
            continue
        method_name, prob_or_det, atlas_name, n_regions = a.split("_")
        print(prob_or_det)
        name = atlas_name.upper()
        if display_regions:
            name = name + " " + n_regions
        if not atlas_only:
            if prob_or_det == "prob":
                dic[a] = "ProbSRM (" + name + ")"
            else:
                dic[a] = "DetSRM (" + name + ")"
        else:
            dic[a] = name
    return {**NAMES, **dic}
