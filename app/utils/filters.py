import markdown as md
from markupsafe import Markup

def markdown(text):
    """Convert Markdown to HTML."""
    if text:
        return Markup(md.markdown(text, extensions=['extra', 'nl2br']))
    return ""