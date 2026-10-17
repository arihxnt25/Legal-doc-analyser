"""Clause-level risk review: turns detected (and *missing*) clauses into
prioritised findings with plain-English explanations and evidence.

Risk in contracts is perspective-dependent (a non-compete binds one side and
protects the other), so findings are phrased as "needs lawyer review" flags,
mirroring CUAD's own framing ("parts of this contract that should be reviewed
by a lawyer"). This is decision support, not legal advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SEVERITY_POINTS = {"HIGH": 18, "MEDIUM": 8, "LOW": 3, "INFO": 0}


@dataclass
class Finding:
    severity: str
    title: str
    explanation: str
    category: str
    kind: str  # "present" | "missing" | "combination"
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# category -> (severity, title, explanation) when the clause IS present
PRESENT_RULES: dict[str, tuple[str, str, str]] = {
    "Uncapped Liability": ("HIGH", "Uncapped liability",
                           "At least one party's liability is not limited. Exposure could exceed the contract value."),
    "Non-Compete": ("HIGH", "Non-compete restriction",
                    "Restricts a party from competing in a market or with the counterparty. Check scope, "
                    "geography and duration for enforceability."),
    "Ip Ownership Assignment": ("HIGH", "IP ownership is assigned",
                                "Intellectual property created under the contract transfers to the other party."),
    "Exclusivity": ("MEDIUM", "Exclusivity commitment",
                    "A party commits to deal exclusively, limiting alternative suppliers or customers."),
    "Liquidated Damages": ("MEDIUM", "Liquidated damages",
                           "Pre-agreed damages are payable on breach or termination."),
    "Change Of Control": ("MEDIUM", "Change-of-control trigger",
                          "A merger or acquisition can trigger termination, consent or other rights; relevant "
                          "in M&A due diligence."),
    "Termination For Convenience": ("MEDIUM", "Termination for convenience",
                                    "A party may terminate without cause, so revenue or supply is not secured "
                                    "for the full term."),
    "Most Favored Nation": ("MEDIUM", "Most-favoured-nation clause",
                            "Better terms given to any third party must also be extended here, which "
                            "constrains future pricing."),
    "Minimum Commitment": ("MEDIUM", "Minimum purchase commitment",
                           "A minimum order volume or spend is required regardless of actual need."),
    "Rofr/Rofo/Rofn": ("MEDIUM", "Right of first refusal / offer / negotiation",
                       "The counterparty gets priority rights over future deals or assets."),
    "Irrevocable Or Perpetual License": ("MEDIUM", "Irrevocable or perpetual licence",
                                         "A licence that cannot be revoked or never expires."),
    "Joint Ip Ownership": ("MEDIUM", "Joint IP ownership",
                           "Jointly owned IP is hard to exploit or license without the co-owner's consent."),
    "Unlimited/All-You-Can-Eat-License": ("MEDIUM", "Unlimited-use licence",
                                          "Usage is unlimited, which may undervalue the licensed asset."),
    "Covenant Not To Sue": ("MEDIUM", "Covenant not to sue",
                            "A party gives up the right to bring claims or challenge IP."),
    "Revenue/Profit Sharing": ("LOW", "Revenue or profit sharing",
                               "Revenue or profit is shared; verify calculation basis and audit rights."),
    "Volume Restriction": ("LOW", "Volume restriction", "Fees rise or consent is needed beyond a volume threshold."),
    "Price Restrictions": ("LOW", "Price restriction", "Limits on raising or lowering prices."),
    "No-Solicit Of Employees": ("LOW", "Employee non-solicit", "Restricts hiring the counterparty's staff."),
    "No-Solicit Of Customers": ("LOW", "Customer non-solicit", "Restricts approaching the counterparty's customers."),
    "Non-Disparagement": ("LOW", "Non-disparagement", "Restricts negative public statements."),
    "Third Party Beneficiary": ("LOW", "Third-party beneficiary",
                                "A non-signatory can enforce rights under the contract."),
    "Source Code Escrow": ("INFO", "Source code escrow", "Source code is deposited with an escrow agent."),
    "Audit Rights": ("INFO", "Audit rights", "A party may audit books, records or premises."),
    "Post-Termination Services": ("INFO", "Post-termination obligations",
                                  "Some obligations survive termination; plan the transition."),
}

# category -> (severity, title, explanation) when the clause is ABSENT
MISSING_RULES: dict[str, tuple[str, str, str]] = {
    "Cap On Liability": ("HIGH", "No limitation-of-liability clause found",
                         "Without a liability cap, damages are limited only by general law."),
    "Governing Law": ("MEDIUM", "No governing-law clause found",
                      "Which jurisdiction's law applies is unclear, raising dispute costs and uncertainty."),
    "Expiration Date": ("MEDIUM", "No clear expiry or term found",
                        "The contract may run indefinitely; check how and when it ends."),
    "Anti-Assignment": ("LOW", "No assignment restriction found",
                        "The counterparty may be able to assign the contract to a third party without consent."),
    "Insurance": ("LOW", "No insurance requirement found",
                  "The counterparty is not required to carry insurance backing its obligations."),
}


def assess(present: dict[str, list[dict]], absent_ok: set[str] | None = None) -> tuple[list[Finding], int]:
    """Return findings (sorted by severity) and a 0-100 review-priority score.

    ``present`` maps category -> list of evidence dicts (text, confidence, ...)
    for every category the classifier detected in the document.
    """
    findings: list[Finding] = []
    absent_ok = absent_ok or set()

    for cat, (sev, title, expl) in PRESENT_RULES.items():
        if cat in present:
            findings.append(Finding(sev, title, expl, cat, "present", present[cat][:3]))

    for cat, (sev, title, expl) in MISSING_RULES.items():
        if cat not in present and cat not in absent_ok:
            findings.append(Finding(sev, title, expl, cat, "missing"))

    # combinations: context changes severity
    if "Renewal Term" in present and "Notice Period To Terminate Renewal" not in present:
        findings.append(Finding("MEDIUM", "Renewal without a clear opt-out notice period",
                                "The contract renews, but no notice period to stop renewal was found. Check "
                                "whether it auto-renews silently.", "Renewal Term", "combination",
                                present["Renewal Term"][:2]))
    elif "Renewal Term" in present:
        findings.append(Finding("LOW", "Renewal term with notice period",
                                "The contract renews; diarise the notice deadline to avoid unwanted renewal.",
                                "Renewal Term", "combination", present["Renewal Term"][:2]))
    if "Uncapped Liability" in present and "Cap On Liability" in present:
        for f in findings:
            if f.category == "Uncapped Liability":
                f.explanation += (" A liability cap also exists, so this is likely a carve-out (e.g. for "
                                  "confidentiality, IP or indemnity); check which claims sit outside the cap.")
    if "Non-Compete" in present and "Competitive Restriction Exception" in present:
        for f in findings:
            if f.category == "Non-Compete":
                f.severity = "MEDIUM"
                f.explanation += " Carve-outs to the restriction were also found, which narrow its scope."
    if "License Grant" in present and "Non-Transferable License" not in present:
        findings.append(Finding("LOW", "Licence transferability not restricted",
                                "A licence is granted but no transfer restriction was found.",
                                "License Grant", "combination", present["License Grant"][:1]))

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    findings.sort(key=lambda f: (order[f.severity], f.kind != "present"))
    score = min(100, sum(SEVERITY_POINTS[f.severity] for f in findings))
    return findings, score


def risk_level(score: int) -> str:
    return "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW"
