"""
IT & Data Job Market Analysis
==============================
Standalone script version of the full analysis pipeline.
Loads the Glassdoor dataset, cleans it, extracts skills via regex,
computes salary distributions, and generates all 5 charts.

Run: python analysis.py
Output: charts/ directory with all 5 PNG files
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/glassdoor_jobs.csv'
CHARTS_DIR  = 'charts'
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Design tokens ──────────────────────────────────────────────────────────────
BG         = '#0D0F1A'
PANEL      = '#13162A'
RULE       = '#1E2138'
PURPLE_HI  = '#7C4DE8'
PURPLE_MID = '#4A3878'
PURPLE_DIM = '#251D3A'
GOLD       = '#F5B731'
WHITE      = '#E8EAF5'
GREY       = '#4B507A'
GREY_MID   = '#6B7299'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'grid.color': RULE, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'xtick.color': GREY_MID,
    'ytick.color': WHITE, 'text.color': WHITE,
})

# ── Font helpers ───────────────────────────────────────────────────────────────
LORA_PATH = '/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf'

def title_font(size=21):
    try:
        return fm.FontProperties(fname=LORA_PATH, size=size, weight=700)
    except Exception:
        return fm.FontProperties(size=size, weight='bold')

def sub_font(size=10):
    return fm.FontProperties(family='sans-serif', size=size, weight='light')

def data_font(size=9):
    return fm.FontProperties(family='monospace', size=size)

def label_font(size=10.5):
    return fm.FontProperties(family='sans-serif', size=size)

def add_header(fig, title, subtitle, yt=0.965, ys=0.918):
    fig.add_artist(plt.Line2D(
        [0.055, 0.20], [yt + 0.019, yt + 0.019],
        transform=fig.transFigure,
        color=PURPLE_HI, linewidth=2.5, solid_capstyle='round'))
    fig.text(0.055, yt, title, fontproperties=title_font(21), color=GOLD, va='top')
    fig.text(0.055, ys, subtitle, fontproperties=sub_font(10), color=GREY, va='top')

def add_footer(fig):
    fig.text(0.055, 0.022,
             'Source: Glassdoor via Kaggle  \u00b7  n=727 postings',
             fontproperties=data_font(8), color=PURPLE_MID, va='bottom')


# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print('Loading data...')
df = pd.read_csv(DATA_PATH)

# Strip Glassdoor rating embedded in company name
df['company_clean'] = df['Company Name'].str.replace(r'\n.*', '', regex=True).str.strip()

# Salary stored in $K — convert to full USD
df['salary_usd'] = df['avg_salary'] * 1000
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Remove outliers and invalid ratings
df = df[(df['salary_usd'] > 20_000) &
        (df['salary_usd'] < 280_000) &
        (df['Rating'] > 0)].copy()

print(f'Clean dataset: {len(df):,} rows')


# ══════════════════════════════════════════════════════════════════════════════
#  2. ROLE BUCKETING
# ══════════════════════════════════════════════════════════════════════════════
def bucket_title(t):
    t = t.lower()
    if 'machine learning' in t or ' ml ' in t: return 'ML Engineer'
    if 'data scientist' in t or 'data science' in t: return 'Data Scientist'
    if 'data engineer' in t:                          return 'Data Engineer'
    if 'data analyst' in t or ('analytics' in t and 'engineer' not in t):
                                                       return 'Data Analyst'
    if 'research scientist' in t:                      return 'Research Scientist'
    return 'Other IT/Data'

df['title_bucket'] = df['Job Title'].apply(bucket_title)


# ══════════════════════════════════════════════════════════════════════════════
#  3. SKILL EXTRACTION — regex NLP on raw description text
# ══════════════════════════════════════════════════════════════════════════════
SKILLS = {
    'Python':          r'\bpython\b',
    'SQL':             r'\bsql\b',
    'Machine Learning': r'machine learning',
    'Spark':           r'\bspark\b',
    'Tableau':         r'tableau',
    'Java':            r'\bjava\b(?!script)',
    'Excel':           r'\bexcel\b',
    'Hadoop':          r'\bhadoop\b',
    'AWS':             r'\baws\b',
    'TensorFlow':      r'tensorflow',
    'Scala':           r'\bscala\b',
    'SAS':             r'\bsas\b',
    'Deep Learning':   r'deep learning',
    'NLP':             r'natural language processing|\bnlp\b',
    'Linux':           r'\blinux\b',
    'scikit-learn':    r'scikit',
    'PyTorch':         r'pytorch',
    'Docker':          r'\bdocker\b',
    'Airflow':         r'\bairflow\b',
    'Power BI':        r'power bi',
}

desc_lower = df['Job Description'].str.lower().fillna('')
for skill, pattern in SKILLS.items():
    df[f'sk_{skill}'] = desc_lower.str.contains(pattern, regex=True).astype(int)

skill_counts = pd.Series(
    {s: df[f'sk_{s}'].sum() for s in SKILLS}
).sort_values(ascending=False)

N = len(df)
print(f'Skills extracted: {len(SKILLS)}')
print(f'Top skill: {skill_counts.index[0]} — {skill_counts.iloc[0]} postings '
      f'({skill_counts.iloc[0]/N*100:.1f}%)')


# ══════════════════════════════════════════════════════════════════════════════
#  4. SUPPORTING DATA
# ══════════════════════════════════════════════════════════════════════════════
top_cos     = df['company_clean'].value_counts()
top_cos     = top_cos[top_cos >= 3].head(12)
sector_map  = df.groupby('company_clean')['Sector'].agg(lambda x: x.mode()[0])
sector_counts = df['Sector'].value_counts().head(8)
sal_by_sec  = df.groupby('Sector')['salary_usd'].median() / 1000

ROLE_COLORS = {
    'Data Analyst':       '#4E6E8A',
    'Data Engineer':      '#3D7A6E',
    'Research Scientist': '#6A5F9E',
    'Data Scientist':     '#9E5A3D',
    'ML Engineer':        '#7C4DE8',
}

def shorten_sector(s):
    return (s.replace('Biotech & Pharmaceuticals', 'Biotech')
             .replace('Business Services', 'Business Svcs')
             .replace('Information Technology', 'IT')
             .replace('Oil, Gas, Energy & Utilities', 'Energy')
             .replace('Aerospace & Defense', 'Defense'))

def bar_col(rank):
    if rank < 3:   return GOLD
    elif rank < 8: return PURPLE_HI
    else:          return PURPLE_MID


# ══════════════════════════════════════════════════════════════════════════════
#  5. CHART 1 — Skill Frequency
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Chart 1 — Skill Frequency...')
top15  = skill_counts.head(15)
pct    = (top15 / N * 100).round(1)
labels = top15.index.tolist()[::-1]
vals   = top15.values[::-1]
pcts   = pct.values[::-1]
total  = len(labels)

fig, ax = plt.subplots(figsize=(12, 7.5))
colors = [bar_col(total - 1 - i) for i in range(total)]
ax.barh(range(total), vals, height=0.62, color=colors, zorder=2, linewidth=0)

for i, (v, p) in enumerate(zip(vals, pcts)):
    c = GOLD if colors[i] == GOLD else (PURPLE_HI if colors[i] == PURPLE_HI else GREY)
    ax.text(v + 5,  i, f'{v}',   fontproperties=data_font(9), color=GREY_MID, va='center', ha='left')
    ax.text(v + 32, i, f'{p}%',  fontproperties=data_font(9), color=c, va='center', ha='left', fontweight='bold')

ax.set_yticks(range(total))
ax.set_yticklabels(labels, fontproperties=label_font(10.5))
ax.set_xlim(0, N * 1.22)
ax.set_xlabel('Job Postings', fontproperties=sub_font(9.5), color=GREY, labelpad=8)
ax.xaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
ax.yaxis.grid(False)
ax.tick_params(axis='x', colors=GREY_MID, labelsize=8.5)
ax.tick_params(axis='y', length=0)

patches = [
    mpatches.Patch(color=GOLD,       label='Top Tier  —  Core Requirement'),
    mpatches.Patch(color=PURPLE_HI,  label='Mid Tier  —  Strong Advantage'),
    mpatches.Patch(color=PURPLE_MID, label='Base Tier  —  Nice To Have'),
]
ax.legend(handles=patches, loc='lower right', fontsize=8.5,
          framealpha=0, labelcolor=GREY_MID, prop=sub_font(8.5))

add_header(fig, 'Skill Frequency',
           '15 most requested skills across 727 job descriptions')
add_footer(fig)
plt.tight_layout(rect=[0, 0.04, 1, 0.90])
plt.savefig(f'{CHARTS_DIR}/01_top_skills.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  6. CHART 2 — Top Hiring Companies
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Chart 2 — Top Hiring Companies...')
co_labels  = top_cos.index.tolist()[::-1]
co_vals    = top_cos.values[::-1]
co_sectors = [sector_map.get(c, '') for c in co_labels]

fig, ax = plt.subplots(figsize=(12, 7))
bar_cols = [GOLD if v == top_cos.max() else PURPLE_HI if v >= 9 else PURPLE_MID
            for v in co_vals]
ax.barh(range(len(co_labels)), co_vals, height=0.6,
        color=bar_cols, zorder=2, linewidth=0)

for i, (v, sec) in enumerate(zip(co_vals, co_sectors)):
    c = GOLD if bar_cols[i] == GOLD else (PURPLE_HI if bar_cols[i] == PURPLE_HI else GREY_MID)
    ax.text(v + 0.15, i, f'{v}',
            fontproperties=data_font(9.5), color=c, va='center', ha='left', fontweight='bold')
    ax.text(v + 1.2, i, f'[{shorten_sector(sec)}]',
            fontproperties=data_font(8), color=PURPLE_MID, va='center', ha='left')

ax.set_yticks(range(len(co_labels)))
ax.set_yticklabels(co_labels, fontproperties=label_font(10.5))
ax.set_xlim(0, top_cos.max() * 1.7)
ax.set_xlabel('Job Postings', fontproperties=sub_font(9.5), color=GREY, labelpad=8)
ax.xaxis.grid(True, color=RULE, linewidth=0.5)
ax.yaxis.grid(False)
ax.tick_params(axis='x', colors=GREY_MID, labelsize=8.5)
ax.tick_params(axis='y', length=0)

add_header(fig, 'Top Hiring Companies',
           'Companies with 3 or more active postings  \u00b7  Sector in brackets')
add_footer(fig)
plt.tight_layout(rect=[0, 0.04, 1, 0.90])
plt.savefig(f'{CHARTS_DIR}/02_top_companies.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  7. CHART 3 — Salary Distribution
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Chart 3 — Salary Distribution...')
role_order = ['Data Analyst', 'Data Engineer', 'Research Scientist',
              'Data Scientist', 'ML Engineer']
role_order = [r for r in role_order if r in df['title_bucket'].unique()]

fig, ax = plt.subplots(figsize=(12, 6.5))
for i, role in enumerate(role_order):
    sub = df[df['title_bucket'] == role]['salary_usd'].dropna()
    if len(sub) < 3:
        continue
    p10, p25, med, p75, p90 = sub.quantile([.10, .25, .50, .75, .90])
    col = ROLE_COLORS.get(role, PURPLE_HI)
    ax.plot([p10/1000, p90/1000], [i, i],
            color=col, alpha=0.22, linewidth=10, solid_capstyle='round', zorder=1)
    ax.plot([p25/1000, p75/1000], [i, i],
            color=col, alpha=0.75, linewidth=16, solid_capstyle='round', zorder=2)
    ax.scatter([med/1000], [i], s=100, color=GOLD, zorder=4, linewidths=0)
    ax.scatter([med/1000], [i], s=32,  color=BG,   zorder=5, linewidths=0)
    ax.text(med/1000, i + 0.42, f'${med/1000:.0f}K',
            fontproperties=data_font(9.5), color=GOLD,
            ha='center', va='bottom', zorder=6)
    ax.text(p90/1000 + 2, i, f'n={len(sub)}',
            fontproperties=data_font(8), color=GREY, va='center', ha='left')

ax.set_yticks(range(len(role_order)))
ax.set_yticklabels(role_order, fontproperties=label_font(11))
ax.set_xlabel('Annual Salary  (USD thousands)',
              fontproperties=sub_font(9.5), color=GREY, labelpad=8)
ax.set_xlim(20, 268)
ax.xaxis.grid(True, color=RULE, linewidth=0.5)
ax.yaxis.grid(False)
ax.tick_params(axis='x', colors=GREY_MID, labelsize=8.5)
ax.tick_params(axis='y', length=0)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${int(x)}K'))
ax.text(0.5, -0.14,
        '\u25cf  Median    \u2501\u2501  25th\u201375th Percentile    \u2500  10th\u201390th Percentile',
        transform=ax.transAxes, ha='center', va='top',
        fontproperties=data_font(8), color=GREY)

add_header(fig, 'Salary Distribution By Role',
           'Median salary with percentile bands  \u00b7  USD annual')
add_footer(fig)
plt.tight_layout(rect=[0, 0.06, 1, 0.90])
plt.savefig(f'{CHARTS_DIR}/03_salary_by_role.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  8. CHART 4 — Demand By Sector
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Chart 4 — Demand By Sector...')
top_sectors = sector_counts.head(8)
sec_labels  = top_sectors.index.tolist()[::-1]
sec_vals    = top_sectors.values[::-1]
sec_sals    = [sal_by_sec.get(s, 0) for s in sec_labels]

fig, ax = plt.subplots(figsize=(12, 6.5))
sec_colors = [GOLD if v == max(sec_vals) else PURPLE_HI if v >= 80 else PURPLE_MID
              for v in sec_vals]
ax.barh(range(len(sec_labels)), sec_vals, height=0.6,
        color=sec_colors, zorder=2, linewidth=0)

for i, (v, sal) in enumerate(zip(sec_vals, sec_sals)):
    c = GOLD if sec_colors[i] == GOLD else (PURPLE_HI if sec_colors[i] == PURPLE_HI else GREY_MID)
    ax.text(v + 1.5, i, f'{v}',
            fontproperties=data_font(9.5), color=c, va='center', ha='left', fontweight='bold')
    ax.text(v + 11, i, f'Median  ${sal:.0f}K',
            fontproperties=data_font(8.5), color=GOLD, va='center', ha='left')

ax.set_yticks(range(len(sec_labels)))
ax.set_yticklabels(sec_labels, fontproperties=label_font(10.5))
ax.set_xlabel('Job Postings', fontproperties=sub_font(9.5), color=GREY, labelpad=8)
ax.set_xlim(0, max(sec_vals) * 1.9)
ax.xaxis.grid(True, color=RULE, linewidth=0.5)
ax.yaxis.grid(False)
ax.tick_params(axis='x', colors=GREY_MID, labelsize=8.5)
ax.tick_params(axis='y', length=0)

add_header(fig, 'Demand By Sector',
           'Posting volume by industry  \u00b7  Median salary shown in gold')
add_footer(fig)
plt.tight_layout(rect=[0, 0.04, 1, 0.90])
plt.savefig(f'{CHARTS_DIR}/04_sector_demand.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  9. CHART 5 — Skill Co-occurrence Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print('Generating Chart 5 — Skill Co-occurrence...')
top10   = skill_counts.head(10).index.tolist()
sk_cols = [f'sk_{s}' for s in top10]
co      = df[sk_cols].T.dot(df[sk_cols])
co.index = co.columns = top10
diag    = np.diag(co.values)
co_pct  = co.values / diag[:, None] * 100
np.fill_diagonal(co_pct, np.nan)

cmap = mcolors.LinearSegmentedColormap.from_list(
    'purple_gold', [PANEL, PURPLE_DIM, PURPLE_MID, PURPLE_HI, '#C8952A', GOLD], N=256)

fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_facecolor(BG)
im = ax.imshow(co_pct, cmap=cmap, vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(top10)))
ax.set_yticks(range(len(top10)))
ax.set_xticklabels(top10, rotation=38, ha='right',
                   fontproperties=sub_font(9.5), color=GREY_MID)
ax.set_yticklabels(top10, fontproperties=label_font(10), color=WHITE)
ax.tick_params(length=0)

for i in range(len(top10)):
    for j in range(len(top10)):
        if i != j:
            val = co_pct[i, j]
            txt_col = BG if val > 65 else WHITE if val > 28 else GREY
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontproperties=data_font(8.5), color=txt_col)

cbar = plt.colorbar(im, ax=ax, shrink=0.58, pad=0.015, aspect=22)
cbar.set_label('% Co-occurrence', fontproperties=sub_font(9), color=GREY)
cbar.ax.tick_params(labelsize=8, colors=GREY)
cbar.outline.set_edgecolor(RULE)
plt.setp(cbar.ax.yaxis.get_ticklabels(), fontproperties=data_font(8), color=GREY)

add_header(fig, 'Skill Co-occurrence',
           'How often skills appear together  \u00b7  Row = base skill, column = paired skill')
add_footer(fig)
plt.tight_layout(rect=[0, 0.04, 1, 0.90])
plt.savefig(f'{CHARTS_DIR}/05_skill_heatmap.png', dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()

print('\n✅ All 5 charts generated successfully')
print(f'Output directory: {os.path.abspath(CHARTS_DIR)}')
