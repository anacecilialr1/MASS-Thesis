"""Shared plotting style."""

import matplotlib.ticker as mticker

AA_STYLE = {
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.linewidth": 0.8, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "figure.dpi": 400,
}

def _patched_call(self, x, pos=None):
    s = mticker.ScalarFormatter._aa_orig_call(self, x, pos)
    if getattr(self, "_useMathText", False):
        s = s.replace(r"\mathdefault{", r"\mathrm{")
    return s

if not hasattr(mticker.ScalarFormatter, "_aa_orig_call"):
    mticker.ScalarFormatter._aa_orig_call = mticker.ScalarFormatter.__call__

mticker.ScalarFormatter.__call__ = _patched_call

def useAA():
    """Apply the style"""
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    from matplotlib.ticker import FuncFormatter
    
    # clear cache if needed
    #from matplotlib import get_cachedir; print(get_cachedir())
    #rm -rf ~/.cache/matplotlib/
    
    # directory of the Cormorant Garamond font 
    font_dir = "/Users/anacecilialr/Downloads/Cormorant_Garamond/static/CormorantGaramond-Medium-renamed.ttf"
    fp = font_manager.FontProperties(fname=font_dir)
    font_fam = fp.get_family()
    font_name = fp.get_name()
    font_weight = fp.get_weight()
    
    font_manager.fontManager.addfont(font_dir)
    #print(f"Succesfully added {font_name}, {font_weight} from the {font_fam} as a font")

    plt.rcParams["font.family"] = font_fam
    plt.rcParams["font.sans-serif"] = font_name
    plt.rcParams.update(AA_STYLE)
    
    return AA_STYLE

# Use this to rename the static font files (because different styles have the same name

##from fontTools.ttLib import TTFont
##
##path = "/Users/anacecilialr/Downloads/Cormorant_Garamond/static/CormorantGaramond-Medium.ttf"
##new_name = "Cormorant Garamond Medium"
##
##font = TTFont(path)
##name_table = font["name"]
##
### nameIDs: 1=Family, 2=Subfamily, 4=Full name, 6=PostScript name, 16=Typographic Family
##for record in name_table.names:
##    if record.nameID in (1, 4, 16):
##        record.string = new_name.encode(record.getEncoding()) if isinstance(new_name, str) else new_name
##
##font.save(path.replace(".ttf", "-renamed.ttf"))
