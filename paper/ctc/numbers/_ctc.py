"""Shared helpers of tables.py and si_tables.py: read data/facts.json, print facts, write pandoc pipe tables + CSV."""
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = json.loads((HERE / 'data/facts.json').read_text())
META = F.pop('_meta')
ORDER = F['s.order']['value']
KEYS = [k for k in ORDER if k != 'brcoc']
COC = [k for k in KEYS if F[f'x.{k}.observed']['value'] == 'cocrystal']
LEVN = {'PBE': 'PBE', 'PBE0': 'PBE0', 'PBE50': 'PBE50', 'PBE0+MP2': 'PBE0 + MP2', 'PBE0+SCS_MP2': 'PBE0 + SCS-MP2',
        'PBE0+wB97X_D3BJ': 'PBE0 + ωB97X-D3(BJ)'}
USED = set()


def v(key):
    USED.add(key)
    return F[key]['value']


def t(key, default='–'):
    """the fact's printed text; '–' when the fact is absent or None."""
    if key not in F or F[key]['value'] is None:
        return default
    USED.add(key)
    return F[key]['text'] if not isinstance(F[key]['text'], list) else ', '.join(F[key]['text'])


def n_of(key):
    USED.add(key)
    return F[key]['text_of']


def env(lo, hi):
    return f'[{t(lo)}, {t(hi)}]'.replace('+', '')


def bond(s):
    return s.replace('...', '···').replace('-', '–') if s else '–'


def write(path, header, align, rows, caption, label, notes=''):
    """pandoc pipe table (house style: caption line with a crossref label, notes paragraph) + the same cells as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = '|' + '|'.join({'l': ':---', 'r': '---:', 'c': ':---:'}[a] for a in align) + '|'
    md = ['| ' + ' | '.join(header) + ' |', sep] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    md += ['', f'Table: {caption} {{#tbl:{label}}}', '']
    if notes:
        md += [notes, '']
    path.with_suffix('.md').write_text('\n'.join(md))
    plain = lambda s: re.sub(r'\^([a-z*]+)\^', r'[\1]', str(s)).replace('−', '-').replace('···', '...').replace('–', '-').replace('*', '')
    with path.with_suffix('.csv').open('w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow([plain(h) for h in header])
        w.writerows([[plain(c) for c in r] for r in rows])
    return path.with_suffix('.md')


def used_report(path):
    """facts each table printed, for the numbers audit (one key per line)."""
    Path(path).write_text('\n'.join(sorted(USED)) + '\n')
