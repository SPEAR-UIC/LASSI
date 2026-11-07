"""Console text formatting and notebook-aware display helpers."""


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def show_markdown(text):
    """Render text as Markdown in a Jupyter notebook; print it in a plain terminal."""
    try:
        from IPython import get_ipython
        from IPython.display import display, Markdown
        if get_ipython() is not None:
            display(Markdown(text))
            return
    except ImportError:
        pass
    print(text)
