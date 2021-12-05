import bs4 as bs
import dash_html_components as html

navbar_html = '<nav class="navbar navbar-expand-lg navbar-light bg-light"><a class="navbar-brand" href="index.html">Protein Structure Similarity<span class="sr-only">(current)</span></a><button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavAltMarkup" aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button><div class="collapse navbar-collapse" id="navbarNavAltMarkup"><div class="navbar-nav"><a class="nav-item nav-link" href="about.html">About</a><a class="nav-item nav-link" href="instructions.html">Instructions</a><a class="nav-item nav-link" href="https://protein-explorer.herokuapp.com/">Protein Explorer</a><a class="nav-item nav-link" href="details.html">Protein Details</a></div></div></nav>'


def convert_html_to_dash(el,style = None):
    CST_PERMITIDOS =  {'nav', 'div','span','a','hr','br','p','b','i','u','s','h1','h2','h3','h4','h5','h6','ol','ul','li',
                        'em','strong','cite','tt','pre','small','big','center','blockquote','address','font','img',
                        'table','tr','td','caption','th','textarea','option'}
    def __extract_style(el):
        if not el.attrs.get("style"):
            return None
        return {k.strip():v.strip() for k,v in [x.split(": ") for x in el.attrs["style"].split(";")]}

    if type(el) is str:
        return convert_html_to_dash(bs.BeautifulSoup(el,'html.parser'))
    if type(el) == bs.element.NavigableString:
        return str(el)
    else:
        name = el.name
        style = __extract_style(el) if style is None else style
        contents = [convert_html_to_dash(x) for x in el.contents]
        if name.title().lower() not in CST_PERMITIDOS:        
            return contents[0] if len(contents)==1 else html.Div(contents)
        return getattr(html,name.title())(contents,style = style)


def parse_html_to_dash(html):
    html_parsed = bs.BeautifulSoup(html)
    dash_html = convert_html_to_dash(html_parsed)
    return dash_html

navbar_dash = parse_html_to_dash(navbar_html) 

if __name__ == '__main__':
    navbar_dash = parse_html_to_dash(navbar_html) 
    print("Done")
