"""Presentation helpers; inspection state and inference stay in app.py."""
from html import escape
from pathlib import Path
from base64 import b64encode

import streamlit as st

APP_TITLE = "Multilayer PCB Defect Detector"


@st.cache_data(show_spinner=False)
def font_styles():
    directory = Path(__file__).with_name('assets') / 'fonts'
    faces = [('Barlow', 'Barlow-Regular.ttf', 400),
             ('Barlow Condensed', 'BarlowCondensed-SemiBold.ttf', 600)]
    return ''.join(
        f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
        f"font-display:swap;src:url(data:font/ttf;base64,{b64encode((directory / filename).read_bytes()).decode('ascii')}) format('truetype');}}"
        for family, filename, weight in faces
    )


def apply_theme():
    st.html(f"<style>{font_styles()}{Path(__file__).with_name('studio.css').read_text(encoding='utf-8')}</style>")


def circuit_board():
    """Decorative routing diagram, not a representation of inspection data."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 220" fill="none">
      <style>
        .trace-base { color: #456451; } .trace-pads { color: #dfa979; }
        .trace-signal { color: #88d6aa; stroke-dasharray: 260; animation: connect 1.2s cubic-bezier(.16,1,.3,1) both; }
        @keyframes connect { from { stroke-dashoffset: 260; } to { stroke-dashoffset: 0; } }
        @media (prefers-reduced-motion: reduce) { .trace-signal { animation: none; } }
      </style>
      <g class="trace-base" stroke="currentColor" stroke-width="2">
        <path d="M0 32H92L132 72H184M0 64H72L112 104H184M0 188H72L120 140H184
                 M440 36H340L308 68H260M440 80H352L320 112H260
                 M440 184H348L316 152H260M220 0V40M236 180V220
                 M16 220V164L64 116V0M408 0V140L376 172V220"/>
        <rect x="184" y="56" width="76" height="124" rx="8"/>
        <rect x="200" y="72" width="44" height="92" rx="3"/>
        <path d="M194 48V56M210 48V56M226 48V56M242 48V56
                 M194 180V188M210 180V188M226 180V188M242 180V188
                 M176 84H184M176 100H184M176 116H184M176 132H184M176 148H184
                 M260 84H268M260 100H268M260 116H268M260 132H268M260 148H268"/>
      </g>
      <g class="trace-pads" stroke="currentColor" stroke-width="2">
        <circle cx="92" cy="32" r="5"/><circle cx="72" cy="64" r="5"/>
        <circle cx="72" cy="188" r="5"/><circle cx="340" cy="36" r="5"/>
        <circle cx="352" cy="80" r="5"/><circle cx="348" cy="184" r="5"/>
        <circle cx="64" cy="116" r="5"/><circle cx="408" cy="140" r="5"/>
      </g>
      <path class="trace-signal" d="M0 64H72L112 104H184" stroke="currentColor" stroke-width="3"/>
    </svg>'''
    # Streamlit's HTML sanitizer strips inline SVG; image data stays inert.
    encoded = b64encode(svg.encode('utf-8')).decode('ascii')
    return f'<img class="pcb-routing" src="data:image/svg+xml;base64,{encoded}" alt="" aria-hidden="true" />'


def studio_header(mode):
    label = escape(mode)
    st.html(f'''<div class="studio-navigation">
      <a class="studio-wordmark" href="#">PCB Detector</a>
      <nav aria-label="Page navigation"><a href="#inspection-workspace">Workspace</a>
        <a href="#inspection-scope">Scope</a></nav>
    </div>
    <div class="studio-header">
      <div class="studio-heading"><h1>Multilayer PCB<br>Defect Detector</h1></div>
      {layer_stack()}
      <div class="hero-bottom"><p>Optical inspection.<br>Traceable scan reports.</p>
        <a class="workspace-link" href="#inspection-workspace">Open inspection workspace</a>
      </div>
    </div>
    <div class="studio-capabilities"><span>{label}</span><span>PNG / JPEG input</span><span>Mode-specific trained models</span></div>''')


def layer_stack():
    """Wireframe PCB layer geometry inspired by the reference's hero linework."""
    layers = []
    for offset in (0, 36, 72, 108):
        layers.append(f'''<g transform="translate(0 {offset})">
          <path d="M80 40L530 0L860 170L410 210Z"/>
          <path d="M170 32L500 190M260 24L590 182M350 16L680 174M440 8L770 166
                   M146 74L596 34M212 108L662 68M278 142L728 102M344 176L794 136"/>
        </g>''')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 350" fill="none">
      <g stroke="#456451" stroke-width="1">{''.join(layers)}</g>
      <g stroke="#dfa979" stroke-width="1.5"><path d="M410 210V318M530 0V108M860 170V278"/>
        <circle cx="410" cy="210" r="5"/><circle cx="530" cy="108" r="5"/><circle cx="860" cy="278" r="5"/></g>
    </svg>'''
    encoded = b64encode(svg.encode('utf-8')).decode('ascii')
    return f'<img class="layer-stack" src="data:image/svg+xml;base64,{encoded}" alt="" aria-hidden="true" />'


def empty_preview():
    st.html(f'''<div class="empty-preview">{circuit_board()}
      <h3>The inspection canvas</h3>
      <p>Upload an image or choose a validation sample to bring it into the workspace.</p>
      <div class="preview-formats">PNG / JPEG · Up to 15 MB</div>
    </div>''')


def defect_legend(classes):
    chips = ''.join(f'<span>{escape(name.replace("_", " "))}</span>' for name in sorted(classes))
    st.html(f'<div class="defect-legend">{chips}</div>')
