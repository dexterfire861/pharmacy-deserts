"""
Shared design system styles for Streamlit pages.

This module centralizes typography and color tokens so all pages use the same
baseline visual language before page-specific layout styling is applied.
"""
from __future__ import annotations

import streamlit as st


def apply_global_design_system(
    *,
    max_width_px: int | None = None,
    top_padding_rem: float = 1.0,
    bottom_padding_rem: float = 1.5,
) -> None:
    """
    Inject global CSS tokens and typography rules.

    Design spec baseline:
    - Font: Helvetica
    - Palette: lime/black/light gray/white
    - Type scale: header 64, sub header 36, body 16
    """
    max_width_rule = (
        f"max-width: {max_width_px}px; width: 100%;"
        if max_width_px is not None
        else "max-width: 100% !important; width: 100%;"
    )

    st.markdown(
        f"""
        <style>
        :root {{
            --pd-lime: #99F200;
            --pd-black: #000000;
            --pd-gray: #EEEEEE;
            --pd-white: #FFFFFF;
            --pd-font: Helvetica, Arial, sans-serif;
            --pd-type-header: 64px;
            --pd-type-subheader: 36px;
            --pd-type-body: 16px;
        }}

        html, body, [class*="css"], [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
        [data-testid="stText"], [data-testid="stFileUploader"], button,
        input, textarea, select, label, p, li, h1, h2, h3, h4, h5, h6 {{
            font-family: var(--pd-font) !important;
        }}
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons,
        [class^="material-symbols"],
        [class*=" material-symbols"],
        [class^="material-icons"],
        [class*=" material-icons"] {{
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            font-style: normal !important;
            font-weight: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            display: inline-block !important;
            white-space: nowrap !important;
            direction: ltr !important;
        }}
        * {{
            accent-color: var(--pd-black) !important;
        }}

        .stApp {{
            background: var(--pd-white);
            color: var(--pd-black);
        }}
        header[data-testid="stHeader"] {{
            display: none !important;
        }}
        [data-testid="stToolbar"] {{
            display: none !important;
        }}
        button[aria-label="Open sidebar"],
        button[title="Open sidebar"],
        [data-testid="stBaseButton-headerNoPadding"] {{
            display: none !important;
        }}
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"] {{
            display: none !important;
        }}

        .block-container {{
            {max_width_rule}
            max-width: none !important;
            padding-top: {top_padding_rem}rem;
            padding-bottom: {bottom_padding_rem}rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 100% !important;
            width: 100% !important;
        }}

        p, li, label, [data-testid="stMarkdownContainer"] p {{
            font-size: var(--pd-type-body);
            font-weight: 400;
            color: var(--pd-black);
        }}

        h1 {{
            font-size: clamp(2.5rem, 6vw, var(--pd-type-header));
            line-height: 1.02;
            font-weight: 500;
            letter-spacing: -0.03em;
            color: var(--pd-black);
        }}

        h2, h3 {{
            font-size: clamp(1.8rem, 4vw, var(--pd-type-subheader));
            line-height: 1.1;
            font-weight: 500;
            letter-spacing: -0.02em;
            color: var(--pd-black);
        }}

        .pd-state-pill {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            font-size: 1rem;
            line-height: 1;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.01em;
        }}

        .pd-state-pill.active {{
            background: var(--pd-lime);
            color: var(--pd-black);
            border: 1px solid var(--pd-lime);
        }}

        .pd-state-pill.inactive {{
            background: var(--pd-white);
            color: var(--pd-black);
            border: 1px solid rgba(153, 242, 0, 0.75);
        }}

        .pd-state-pill.neither {{
            background: var(--pd-black);
            color: var(--pd-white);
            border: 1px solid var(--pd-black);
        }}

        div.stButton > button,
        div.stDownloadButton > button {{
            background: var(--pd-black);
            color: var(--pd-white);
            border: 1px solid var(--pd-black);
            border-radius: 999px;
            min-height: 36px;
            font-size: var(--pd-type-body);
            font-weight: 500;
        }}

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {{
            color: var(--pd-lime);
            border-color: var(--pd-lime);
        }}

        div[data-testid="stRadio"] [role="radiogroup"] {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }}

        div[data-testid="stRadio"] [role="radiogroup"] > label {{
            background: var(--pd-white) !important;
            border: 1.25px solid rgba(153, 242, 0, 0.78) !important;
            border-radius: 999px !important;
            padding: 0.28rem 1.05rem !important;
            min-height: 40px;
            display: inline-flex !important;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
        }}

        div[data-testid="stRadio"] [role="radiogroup"] > label:has(input[type="radio"]:checked),
        div[data-testid="stRadio"] [role="radiogroup"] > label:has([aria-checked="true"]) {{
            background: var(--pd-lime) !important;
            border-color: var(--pd-lime) !important;
            color: var(--pd-black) !important;
        }}

        div[data-testid="stRadio"] [role="radiogroup"] > label p {{
            margin: 0 !important;
            color: var(--pd-black) !important;
            font-weight: 500 !important;
            text-align: center !important;
            line-height: 1.12 !important;
        }}

        div[data-testid="stRadio"] [role="radiogroup"] > label > div:last-child {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            min-height: 100% !important;
        }}
        div[data-testid="stRadio"] [role="radiogroup"] > label [data-testid="stMarkdownContainer"] {{
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}

        div[data-testid="stRadio"] [role="radiogroup"] > label input[type="radio"],
        div[data-testid="stRadio"] [role="radiogroup"] > label [role="radio"] {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            opacity: 0 !important;
            margin: 0 !important;
        }}

        div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
            min-height: 18px !important;
            height: 18px !important;
            display: flex !important;
            align-items: center !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{
            height: 4px !important;
            border-radius: 999px !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
            width: 16px !important;
            height: 16px !important;
            border-radius: 999px !important;
            background: var(--pd-white) !important;
            border: 2px solid var(--pd-black) !important;
            box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.08) !important;
            margin-top: -6px !important;
            z-index: 3 !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]::before,
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]::after {{
            content: none !important;
        }}
        div[data-testid="stSlider"] [data-testid="stThumbValue"],
        div[data-testid="stSlider"] [data-testid="stThumbValue"] * {{
            display: block !important;
            color: var(--pd-black) !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            line-height: 1 !important;
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
