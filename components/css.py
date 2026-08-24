"""Komponen CSS Kustom untuk Streamlit."""

import streamlit as st

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text-primary: #fafafa;
            --text-secondary: rgba(229,229,229,0.82);
            --text-body: #f5f5f5;
            --text-muted: rgba(212,212,212,0.78);
            --text-tertiary: #a3a3a3;
            --text-quaternary: #737373;
            --surface-1: rgba(10,10,10,0.45);
            --surface-2: rgba(23,23,23,0.55);
            --border-subtle: rgba(163,163,163,0.14);
            --border-strong: rgba(255,255,255,0.22);
            --radius-sm: 10px;
            --radius-md: 14px;
            --radius-lg: 20px;
            --transition-fast: 150ms ease;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            color: var(--text-secondary);
            margin: 0 0 1rem 0;
            line-height: 1.55;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            color: var(--text-primary);
            border: 1px solid var(--border-strong);
            border-radius: 999px;
            padding: 0.28rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .section-card {
            background: var(--surface-2);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            transition: transform var(--transition-fast), border-color var(--transition-fast);
        }
        .section-card:hover {
            transform: translateY(-2px);
            border-color: var(--border-strong);
        }
        .section-step {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-primary);
            margin-bottom: 0.35rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 650;
            margin: 0 0 0.35rem 0;
        }
        .section-desc {
            font-size: 0.92rem;
            color: var(--text-muted);
            margin: 0 0 0.75rem 0;
        }
        .meta-label {
            font-size: 0.75rem;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.15rem;
        }
        .meta-value {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            word-break: break-word;
        }
        .empty-state {
            text-align: center;
            padding: 2.5rem 1.5rem;
            border: 1px dashed rgba(163,163,163,0.28);
            border-radius: var(--radius-md);
            background: rgba(10,10,10,0.35);
            margin: 1rem 0 1.5rem 0;
        }
        .empty-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
        .empty-title { font-size: 1.05rem; font-weight: 650; margin-bottom: 0.25rem; }
        .empty-desc { font-size: 0.92rem; color: var(--text-muted); }
        .result-card {
            background: linear-gradient(160deg, rgba(23,23,23,0.92), rgba(10,10,10,0.82));
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-lg);
            padding: 2.75rem 2rem;
            margin: 1rem 0 1.25rem 0;
            text-align: center;
            box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        }
        .result-inner {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .result-emoji {
            font-size: 4.5rem;
            line-height: 1;
            margin-bottom: 0.65rem;
            filter: drop-shadow(0 4px 12px rgba(0,0,0,0.25));
        }
        .result-label {
            font-size: 2.1rem;
            font-weight: 700;
            margin: 0 0 0.85rem 0;
            text-transform: capitalize;
            letter-spacing: -0.01em;
        }
        .result-conf-label {
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--text-tertiary);
            margin-bottom: 0.35rem;
        }
        .result-confidence {
            font-size: 4.25rem;
            font-weight: 800;
            margin: 0;
            line-height: 1;
            background: linear-gradient(135deg, #ffffff, #d4d4d4, #a3a3a3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .result-dominance {
            font-size: 0.82rem;
            color: var(--text-tertiary);
            margin: 0 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 650;
        }
        .result-rank-note {
            font-size: 0.92rem;
            color: var(--text-muted);
            margin: 0.85rem 0 0 0;
            line-height: 1.45;
        }
        .result-margin {
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid var(--border-strong);
            font-size: 0.82rem;
            color: var(--text-primary);
            font-weight: 600;
        }
        .top3-card {
            background: var(--surface-1);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 0.9rem 1rem;
            text-align: center;
            min-height: 118px;
            transition: transform var(--transition-fast), border-color var(--transition-fast);
        }
        .top3-card:hover {
            transform: translateY(-2px);
            border-color: var(--border-strong);
        }
        .top3-rank {
            font-size: 0.72rem;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .top3-emoji { font-size: 1.6rem; margin: 0.15rem 0; }
        .top3-label { font-size: 1rem; font-weight: 650; text-transform: capitalize; }
        .top3-pct {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 0.15rem;
        }
        .top3-conf-label {
            font-size: 0.68rem;
            color: var(--text-quaternary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .transcript-card {
            background: linear-gradient(145deg, rgba(23,23,23,0.85), rgba(10,10,10,0.78));
            border: 1px solid var(--border-strong);
            border-left: 4px solid var(--text-primary);
            border-radius: var(--radius-md);
            padding: 1.1rem 1.35rem;
            margin: 0.5rem 0 1rem 0;
        }
        .transcript-head {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-primary);
            margin-bottom: 0.45rem;
        }
        .transcript-text {
            font-size: 1.05rem;
            color: var(--text-body);
            line-height: 1.55;
            font-style: italic;
        }
        .transcript-empty {
            font-size: 0.95rem;
            color: rgba(163,163,163,0.85);
            font-style: italic;
        }
        .segment-card {
            background: var(--surface-2);
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--emotion-color, #a3a3a3);
            border-radius: var(--radius-sm);
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
            transition: transform var(--transition-fast), border-color var(--transition-fast);
        }
        .segment-card:hover {
            transform: translateY(-2px);
        }
        .segment-time {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-tertiary);
            margin-bottom: 0.3rem;
        }
        .segment-text {
            font-size: 0.95rem;
            color: var(--text-body);
            font-style: italic;
            margin-bottom: 0.45rem;
        }
        .segment-emotion-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 650;
        }
        .prob-row-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.92rem;
            margin-bottom: 0.2rem;
        }
        .prob-bar-wrap {
            background: rgba(38,38,38,0.65);
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
            margin-bottom: 0.85rem;
        }
        .prob-bar-fill {
            height: 10px;
            border-radius: 999px;
            transition: width var(--transition-fast);
        }
        .sidebar-pill {
            display: inline-block;
            background: rgba(38,38,38,0.75);
            border: 1px solid var(--border-subtle);
            border-radius: 999px;
            padding: 0.18rem 0.62rem;
            margin: 0.12rem 0.18rem 0.12rem 0;
            font-size: 0.78rem;
            text-transform: capitalize;
        }
        .sidebar-header {
            background: linear-gradient(145deg, rgba(255,255,255,0.14), rgba(255,255,255,0.05));
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-md);
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 0.85rem;
            text-align: center;
        }
        .sidebar-header-icon {
            font-size: 1.75rem;
            margin-bottom: 0.25rem;
        }
        .sidebar-header-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.01em;
        }
        .sidebar-header-sub {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin: 0.25rem 0 0 0;
            line-height: 1.4;
        }
        .sidebar-stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin-bottom: 0.85rem;
        }
        .sidebar-stat-card {
            background: var(--surface-1);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 0.65rem 0.7rem;
        }
        .sidebar-stat-icon {
            font-size: 1rem;
            margin-bottom: 0.15rem;
        }
        .sidebar-stat-label {
            font-size: 0.68rem;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.1rem;
        }
        .sidebar-stat-value {
            font-size: 0.82rem;
            font-weight: 650;
            color: var(--text-body);
            line-height: 1.25;
        }
        .sidebar-status-card {
            border-radius: var(--radius-sm);
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }
        .sidebar-status-ok {
            background: rgba(255,255,255,0.08);
            border: 1px solid var(--border-strong);
        }
        .sidebar-status-fail {
            background: rgba(248,113,113,0.1);
            border: 1px solid rgba(248,113,113,0.28);
        }
        .sidebar-status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .sidebar-status-dot.ok {
            background: var(--text-primary);
            box-shadow: 0 0 8px rgba(255,255,255,0.55);
        }
        .sidebar-status-dot.fail {
            background: #f87171;
            box-shadow: 0 0 8px rgba(248,113,113,0.55);
        }
        .sidebar-status-text {
            font-size: 0.84rem;
            font-weight: 650;
            color: var(--text-primary);
        }
        .sidebar-status-sub {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 0.05rem;
        }
        .sidebar-section-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--text-primary);
            margin: 0.15rem 0 0.55rem 0;
        }
        .sidebar-emotion-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.45rem;
            margin-bottom: 0.85rem;
        }
        .sidebar-emotion-item {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            background: var(--surface-1);
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--emotion-color, #a3a3a3);
            border-radius: var(--radius-sm);
            padding: 0.42rem 0.55rem;
            font-size: 0.78rem;
            text-transform: capitalize;
            transition: transform var(--transition-fast);
        }
        .sidebar-emotion-item:hover {
            transform: translateY(-2px);
        }
        .sidebar-emotion-emoji { font-size: 1rem; line-height: 1; }
        .sidebar-emotion-id {
            font-size: 0.65rem;
            color: var(--text-quaternary);
            margin-left: auto;
        }
        .sidebar-history-list {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            margin-bottom: 0.85rem;
        }
        .sidebar-history-item {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            background: var(--surface-1);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 0.45rem 0.6rem;
            transition: transform var(--transition-fast);
        }
        .sidebar-history-item:hover {
            transform: translateY(-2px);
        }
        .sidebar-history-emoji { font-size: 1.15rem; line-height: 1; }
        .sidebar-history-label {
            font-size: 0.8rem;
            font-weight: 650;
            color: var(--text-body);
            text-transform: capitalize;
        }
        .sidebar-history-meta {
            font-size: 0.68rem;
            color: var(--text-quaternary);
        }
        .sidebar-howto {
            background: var(--surface-2);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.75rem;
        }
        .sidebar-howto-step {
            display: flex;
            align-items: flex-start;
            gap: 0.55rem;
            margin-bottom: 0.55rem;
        }
        .sidebar-howto-step:last-child { margin-bottom: 0; }
        .sidebar-howto-num {
            background: rgba(255,255,255,0.16);
            color: var(--text-primary);
            border-radius: 999px;
            width: 1.35rem;
            height: 1.35rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 700;
            flex-shrink: 0;
        }
        .sidebar-howto-text {
            font-size: 0.78rem;
            color: var(--text-secondary);
            line-height: 1.35;
            padding-top: 0.05rem;
        }
        .sidebar-divider {
            border: none;
            border-top: 1px solid var(--border-subtle);
            margin: 0.65rem 0;
        }
        .dash-status-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .dash-status-card {
            background: var(--surface-2);
            border: 1px solid var(--border-subtle);
            border-left: 3px solid #f87171;
            border-radius: var(--radius-md);
            padding: 1rem 1.15rem;
            display: flex;
            align-items: center;
            gap: 0.85rem;
            transition: transform var(--transition-fast), border-color var(--transition-fast);
        }
        .dash-status-card.ok { border-left-color: var(--text-primary); }
        .dash-status-card:hover { transform: translateY(-2px); }
        .dash-status-card--full { grid-column: 1 / -1; }
        .dash-status-icon { font-size: 1.5rem; line-height: 1; flex-shrink: 0; }
        .dash-status-body { display: flex; flex-direction: column; gap: 0.15rem; min-width: 0; }
        .dash-status-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-tertiary);
        }
        .dash-status-value {
            font-size: 1.05rem;
            font-weight: 650;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }
        .dash-activity-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin: 0.5rem 0 1rem 0;
        }
        .dash-activity-row {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            background: var(--surface-1);
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--emotion-color, #a3a3a3);
            border-radius: var(--radius-sm);
            padding: 0.65rem 0.9rem;
            transition: transform var(--transition-fast);
        }
        .dash-activity-row:hover { transform: translateY(-2px); }
        .dash-activity-emoji { font-size: 1.35rem; line-height: 1; flex-shrink: 0; }
        .dash-activity-main { flex: 1; min-width: 0; }
        .dash-activity-label { font-size: 0.92rem; font-weight: 650; color: var(--text-body); text-transform: capitalize; }
        .dash-activity-meta {
            font-size: 0.72rem;
            color: var(--text-quaternary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .dash-activity-conf {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
            flex-shrink: 0;
        }
        .status-ok { color: var(--text-primary); font-weight: 650; }
        .status-fail { color: #f87171; font-weight: 650; }
        a[data-testid="stTopNavLink"],
        a[data-testid="stSidebarNavLink"] {
            padding: 0.5rem 0.9rem;
        }
        span[data-testid="stIconMaterial"] {
            color: var(--text-tertiary) !important;
        }
        a[data-testid="stTopNavLink"][aria-current="page"] span[data-testid="stIconMaterial"],
        a[data-testid="stSidebarNavLink"][aria-current="page"] span[data-testid="stIconMaterial"] {
            color: var(--text-primary) !important;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(5,5,5,0.98), rgba(0,0,0,0.92));
        }
        div[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #fafafa, #d4d4d4) !important;
            border: none !important;
            color: #0a0a0a !important;
            border-radius: 12px !important;
            padding: 0.72rem 1rem !important;
            font-weight: 650 !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.45) !important;
            transition: box-shadow var(--transition-fast), background var(--transition-fast) !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #e5e5e5, #c7c7c7) !important;
            box-shadow: 0 10px 28px rgba(0,0,0,0.55) !important;
        }
        div.stButton > button:disabled {
            opacity: 0.55 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )