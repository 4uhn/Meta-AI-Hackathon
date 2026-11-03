"""Network flow -> compact `key=value` text.

Numbers are rounded to 3 significant figures: shorter prompts, and near-identical flood flows collapse to
one string so dedupe can keep test strings out of training. Missing values become `na`; on the destination
side that usually means "no reply packets", which is informative, so those columns are kept.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SERIALIZER_VERSION = "v1"

# Canonical labels, in the fixed order used everywhere (prompts, matrices, configs).
LABELS = [
    "benign",
    "udp_flood",
    "icmp_flood",
    "syn_flood",
    "http_flood",
    "slowrate_dos",
    "syn_scan",
    "tcp_connect_scan",
    "udp_scan",
]
RAW_TO_CANONICAL = {
    "Benign": "benign",
    "UDPFlood": "udp_flood",
    "ICMPFlood": "icmp_flood",
    "SYNFlood": "syn_flood",
    "HTTPFlood": "http_flood",
    "SlowrateDoS": "slowrate_dos",
    "SYNScan": "syn_scan",
    "TCPConnectScan": "tcp_connect_scan",
    "UDPScan": "udp_scan",
}

# Columns removed before anything else, with the reason (recorded in stats.json).
DROP_COLUMNS = {
    "Unnamed: 0": "row index",
    "Seq": "Argus record sequence number (row order)",
    "Offset": "byte offset in the capture file (row order)",
    "RunTime": "duplicate of Dur",
    "Mean": "duplicate of Dur for single-record flows",
    "Sum": "duplicate of Dur for single-record flows",
    "Min": "duplicate of Dur for single-record flows",
    "Max": "duplicate of Dur for single-record flows",
    "SrcTCPBase": "TCP initial sequence number (identifier)",
    "DstTCPBase": "TCP initial sequence number (identifier)",
    "sVid": "VLAN id, constant",
    "dVid": "VLAN id, constant, >99% null",
    "Attack Tool": "label leakage",
    "Label": "binary label (derived from Attack Type)",
}

# Serialised fields: key → source column(s). Pairs are rendered "src/dst".
FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("proto", ("Proto",)),
    ("state", ("State",)),
    ("cause", ("Cause",)),
    ("dur", ("Dur",)),
    ("pkts", ("SrcPkts", "DstPkts")),
    ("bytes", ("SrcBytes", "DstBytes")),
    ("rate", ("Rate",)),
    ("msz", ("sMeanPktSz", "dMeanPktSz")),
    ("ttl", ("sTtl", "dTtl")),
    ("tos", ("sTos", "dTos")),
    ("loss", ("SrcLoss", "DstLoss")),
    ("rtt", ("TcpRtt",)),
    ("synack", ("SynAck",)),
    ("ackdat", ("AckDat",)),
    ("win", ("SrcWin", "DstWin")),
]
FEATURE_COLUMNS = [c for _, cols in FIELDS for c in cols]


def fmt_num(x: object) -> str:
    """3 significant figures, no exponent for |x| in [1e-4, 1e6), integers without a decimal point."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "na"
    if isinstance(x, str):
        return x
    v = float(x)
    if v == 0:
        return "0"
    r = float(f"{v:.3g}")
    if r == int(r) and abs(r) < 1e15:
        return str(int(r))
    if 1e-4 <= abs(r) < 1e6:
        return np.format_float_positional(r, trim="-")
    return f"{r:.3g}"


def _fmt_column(s: pd.Series) -> pd.Series:
    """Format a column by formatting each distinct value once (fast on 1.2M rows)."""
    if s.dtype == object:
        return s.fillna("na").astype(str).str.strip().str.lower()
    uniq = pd.unique(s.dropna())
    table = {u: fmt_num(u) for u in uniq}
    return s.map(table).fillna("na")


def serialize_frame(df: pd.DataFrame) -> pd.Series:
    """Serialise every row of `df` (raw column names) to the compact text form."""
    formatted = {c: _fmt_column(df[c]) for c in FEATURE_COLUMNS}
    parts = []
    for key, cols in FIELDS:
        val = formatted[cols[0]]
        for c in cols[1:]:
            val = val + "/" + formatted[c]
        parts.append(key + "=" + val)
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out
