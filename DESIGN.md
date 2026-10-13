---
name: Multilayer PCB Defect Detector
description: A solder-mask workspace for image inspection and traceable reports.
colors:
  bg: '#0b1714'
  surface: '#1b3027'
  raised: '#243e31'
  line: '#456451'
  text: '#edf5ef'
  muted: '#b0c4b8'
  green: '#88d6aa'
  copper: '#dfa979'
  copper-hover: '#ecc09a'
  disabled-bg: '#293e33'
  disabled-text: '#9cb0a3'
typography:
  display:
    fontFamily: '"Barlow", sans-serif'
    fontSize: clamp(3rem, 8vw, 7rem)
    fontWeight: 400
    lineHeight: 0.98
    letterSpacing: -.035em
  title:
    fontFamily: '"Barlow", sans-serif'
    fontSize: 1.75rem
    fontWeight: 400
    letterSpacing: -.025em
  body:
    fontFamily: '"Barlow", sans-serif'
    lineHeight: 1.55
  label:
    fontFamily: '"Barlow", sans-serif'
    fontSize: .85rem
  metric:
    fontFamily: '"Barlow Condensed", sans-serif'
    fontSize: clamp(1.3rem, 2.2vw, 2rem)
rounded:
  chip: 0px
  control: 0px
  tab: 0px
  upload: 0px
  panel: 0px
spacing:
  xs: .5rem
  sm: .75rem
  md: 1rem
  lg: 1.25rem
  xl: 1.5rem
  xxl: 2rem
components:
  button-primary:
    backgroundColor: '{colors.copper}'
    textColor: '{colors.bg}'
    rounded: '{rounded.control}'
    height: 50px
  button-disabled:
    backgroundColor: '{colors.disabled-bg}'
    textColor: '{colors.disabled-text}'
    rounded: '{rounded.control}'
  input:
    rounded: '{rounded.control}'
  tab:
    textColor: '{colors.muted}'
    rounded: '{rounded.tab}'
    padding: .5rem 1.25rem
    height: 50px
  tab-selected:
    backgroundColor: '{colors.copper}'
    textColor: '{colors.bg}'
  defect-chip:
    textColor: '{colors.muted}'
    backgroundColor: '{colors.raised}'
    rounded: '{rounded.chip}'
    padding: .35rem .75rem
  intake:
    rounded: '{rounded.panel}'
    padding: 1.5rem
  metric:
    rounded: '{rounded.upload}'
    backgroundColor: '{colors.raised}'
    padding: 1rem
---

# Design System: Multilayer PCB Defect Detector

## Overview

**Creative North Star: "The Board Workspace"**

The supplied Wix wh-1081 template guides the composition: oversized light sans-serif lettering, wireframe geometry behind the title, sparse navigation, thin dividers, square edges and a broad action band. The detector adapts these devices to PCB layers and the existing solid green/copper palette, rather than importing financial content or claims.

The interface remains an operating tool: the opening action links directly to the real inspection controls. Optical scope, model validation, cached inference, history and report downloads retain their behavior.

## Colors

Use the approximate 60–30–10 hierarchy: deep board green for the dominant ground, supporting green for workspace surfaces, and restrained copper/signal-green accents. Primary actions and selected tabs use copper; focus, active-mode details and chart marks use signal green. All frontend backgrounds are solid.

## Typography

Locally bundled Barlow is the display and body face. The hero title uses weight 400, a 0.98 line-height and a responsive size up to 7rem. At 768px and below it uses clamp(2.1rem, 9.5vw, 4rem). Section headings use regular Barlow with tight tracking. Barlow Condensed remains for tabular metric values. Fonts use font-display: swap.

## Layout

The main container reaches 1600px with 1.75rem desktop side padding. A simple wordmark/navigation row precedes a 490px minimum-height hero. Decorative stacked-board wireframe geometry sits behind the title. The bottom hero row divides supporting text and the copper workspace link into equal columns. A capability strip follows.

Real controls appear under the inspection-workspace anchor. The configuration bar and full-width, three-part workflow navigation lead to the 1.85:1 canvas/intake split. At 768px and below, the hero becomes 360px minimum height, the action band stacks, controls stack, and intake precedes the canvas. Both anchors clear the fixed toolbar by 80px.

## Elevation & Depth

Use solid surface steps and fine borders. Wireframe artwork is flat line geometry. Only the primary inspection action lifts with a soft shadow on hover; avoid decorative panel shadows.

## Shapes

All panels, inputs, tabs, buttons, metrics and chips have square corners. Borders and broad solid bands establish the layout. The layered board diagram is decorative geometry and does not claim to visualize scan results or internal layers.

## Components

Navigation links point to real workspace and scope anchors. The hero workspace link is a broad copper rectangle with a lighter-copper hover state. Workflow tabs span the available width with divider lines and a copper selected state. Inputs and secondary actions retain native Streamlit semantics and 46px minimum heights; the inspection action remains 50px. Supporting panels use solid green fills.

Controls transition over 240ms, fields over 220ms, and tab panels enter over 380ms using cubic-bezier(.16, 1, .3, 1). The decorative trace animation remains 1.2s. Anchor navigation uses native smooth scrolling on Streamlit's main scroll container, with 80px target clearance for the toolbar. Reduced-motion preferences restore immediate scrolling and disable authored animations and transitions. Keyboard focus remains visible.

## Do's and Don'ts

- Do adapt the reference's composition to real detector content.
- Do keep solid green surfaces dominant and accents restrained.
- Do preserve working controls, visible focus and accurate optical scope.
- Don't introduce gradients, rounded dashboard panels or invented performance claims.
- Don't present decorative PCB layers as CT or internal-layer inference.
