import streamlit as st


def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('div[class*="stTextInput"]');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
        elem.style.color = '{font_color}';
        elem.style.fontFamily = '{font_family}';
    </script>
    """
    return st.components.v1.html(html)


TOOL_HIDE = """
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            .css-hi6a2p {padding-top: 0rem;}
        </style>
    """

label = r'''
        $\textsf{
        \large Ask Me 💬
        }$
    '''

title = "♌ Horoscope Virtual Assistant 💬"


# footer = """
#     2015 Flight Delays and Cancellations.
#    """
footer = """Vedic astrology, also known as Jyotish, is an ancient system of astrology that originated in India. The term "Vedic" comes from the Vedas, which are ancient Hindu scriptures. Jyotish means "science of light" in Sanskrit, and Vedic astrology is often considered a spiritual science."""
side_foot = f"<p style='font-size:15px;'>{footer}</p>"


def get_page_conf():
    return st.set_page_config(
        page_title="Virtual Assistant 💬",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'About': "# 2015 Flight Delays and Cancellations."
        }
    )


def get_footer():
    footer = """
        <style>
        a:link , a:visited{
        color: blue;
        background-color: transparent;
        text-decoration: underline;
        }

        a:hover,  a:active {
        color: red;
        background-color: transparent;
        text-decoration: underline;
        }

        .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        }
        </style>
        <div class="footer">
        <p> <a style='display: block; color: black; text-align: center;' href="https://www.google.com/" target="_blank">📊 Contact Analytics Team</a></p>
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
