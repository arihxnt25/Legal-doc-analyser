"""Rule-based extraction of key contract terms.

Deliberately regex-based: dates, money and durations follow rigid formats in
contracts, regexes are exact, fast and auditable, and they complement the
clause classifier (which finds *where* the governing-law clause is; the
regex reads *which* jurisdiction it names).
"""
from __future__ import annotations

import re
from collections import Counter

MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
DATE_RE = re.compile(
    rf"\b(?:(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:{MONTHS}),?\s+\d{{4}}"
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})\b", re.I)
MONEY_RE = re.compile(
    r"(?:(?:US)?\$|USD|EUR|€|£|GBP|INR|Rs\.?|₹)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand|[MBK]))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion)?\s?(?:dollars|USD|euros|rupees)\b", re.I)
NUM_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|" \
            r"forty-five|forty|sixty|ninety|hundred|\d+)"
DURATION_RE = re.compile(
    rf"\b{NUM_WORDS}(?:\s*\(\d+\))?\s+(?:calendar\s+|business\s+|consecutive\s+)?"
    r"(?:days?|weeks?|months?|years?)\b", re.I)
PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?(?:%|percent\b)", re.I)

US_STATES = ("Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|"
             "Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|"
             "Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|"
             "New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
             "Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
             "West Virginia|Wisconsin|Wyoming|District of Columbia")
COUNTRIES = ("England and Wales|England|Scotland|Ireland|India|Singapore|Hong Kong|the People's Republic of "
             "China|China|Japan|Germany|France|Switzerland|the Netherlands|Canada|Ontario|British Columbia|"
             "Australia|Israel|Delaware|the United States")
GOV_LAW_RE = re.compile(
    rf"(?:governed\s+by|construed\s+(?:and\s+enforced\s+)?in\s+accordance\s+with|laws?\s+of)\s+"
    rf"(?:and\s+construed\s+in\s+accordance\s+with\s+)?(?:the\s+)?(?:laws?\s+of\s+)?(?:the\s+)?"
    rf"(?:State\s+of\s+|Commonwealth\s+of\s+|Province\s+of\s+)?({US_STATES}|{COUNTRIES})", re.I)
PARTIES_RE = re.compile(
    r"(?:by\s+and\s+between|between|among)\s*:?\s+(.{3,240}?)\s*(?:\(|,)?\s*-*\s*(?:and|AND)\s*-*\s+(.{3,240}?)"
    r"(?:\s*\(|,\s*a\s|\.\s|\s+\(collectively|\s+WHEREAS|\s+RECITALS|\s+\"the\s|\s{3,}|\n)", re.S | re.I)
_SUFFIX = (r"(?:Incorporated|Inc|Corporation|Corp|L\.L\.C|LLC|Limited|Ltd|LLP|LP|L\.P|PLC|GmbH|AG|S\.A|N\.V|B\.V|"
           r"Company|Co|Pvt|Private)")
# one or more chained suffixes: "Co., Ltd.", "Company Limited", "Pvt. Ltd."
ORG_SUFFIX = re.compile(rf"\b{_SUFFIX}\b\.?(?:,?\s+{_SUFFIX}\b\.?)*", re.I)


def _clean_party(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" ,;:\"'-")
    s = re.split(r"\s*\(|\s*\"|,\s*(?:a|an)\s+[A-Za-z]", s)[0]  # drop defined terms / descriptions
    m = ORG_SUFFIX.search(s)
    if m:  # "ACME INC. 2nd Floor, ..." -> keep up to the company suffix, drop the address
        s = s[:m.end()]
    return s.strip(" ,;:")[:120]


def extract_parties(text: str) -> list[str]:
    head = text[:6000]
    for m in PARTIES_RE.finditer(head):
        parties = [_clean_party(m.group(1)), _clean_party(m.group(2))]
        # skip page headers/footers and fragments that are not names
        if all(2 < len(p) <= 120 and not re.search(r"\bpage\s+\d|^\d", p, re.I) for p in parties):
            return parties
    # fallback: most frequent organisation-like names in the preamble
    orgs = re.findall(r"((?:[A-Z][\w&.,'-]*\s){1,6}" + ORG_SUFFIX.pattern + ")", head)
    return [o.strip() for o, _ in Counter(orgs).most_common(2)]


def extract_governing_law(text: str) -> str | None:
    m = GOV_LAW_RE.search(text)
    return m.group(1).strip() if m else None


def _uniq(items, limit):
    seen, out = set(), []
    for x in items:
        k = re.sub(r"\s+", " ", x.strip())
        if k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
        if len(out) >= limit:
            break
    return out


def extract_entities(text: str, limit: int = 15) -> dict:
    return {
        "parties": extract_parties(text),
        "governing_law": extract_governing_law(text),
        "dates": _uniq(DATE_RE.findall(text), limit),
        "monetary_amounts": _uniq(MONEY_RE.findall(text), limit),
        "durations": _uniq(DURATION_RE.findall(text), limit),
        "percentages": _uniq(PERCENT_RE.findall(text), limit),
    }
