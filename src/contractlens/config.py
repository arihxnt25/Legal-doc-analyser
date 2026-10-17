"""Paths and the CUAD clause taxonomy (41 categories) with review metadata."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("CONTRACTLENS_DATA_DIR", ROOT / "data" / "cuad"))
ARTIFACT_DIR = Path(os.getenv("CONTRACTLENS_ARTIFACT_DIR", ROOT / "artifacts"))
CUAD_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"


@dataclass(frozen=True)
class Category:
    name: str
    group: str
    description: str


# Order matches CUAD v1. Groups follow the CUAD paper's grouping of clause types.
CATEGORIES: list[Category] = [
    Category("Document Name", "General", "The name of the contract"),
    Category("Parties", "General", "The two or more parties who signed the contract"),
    Category("Agreement Date", "General", "The date of the contract"),
    Category("Effective Date", "General", "The date when the contract is effective"),
    Category("Expiration Date", "Term", "On what date will the contract's initial term expire?"),
    Category("Renewal Term", "Term", "Renewal term after the initial term expires (incl. auto-renewal)"),
    Category("Notice Period To Terminate Renewal", "Term", "Notice period required to terminate renewal"),
    Category("Governing Law", "General", "Which state/country's law governs interpretation of the contract?"),
    Category("Most Favored Nation", "Restrictive", "Better terms offered to a third party must be offered to this party"),
    Category("Non-Compete", "Restrictive", "Restriction on competing with the counterparty or in a market"),
    Category("Exclusivity", "Restrictive", "Exclusive dealing commitment"),
    Category("No-Solicit Of Customers", "Restrictive", "Restriction on soliciting the counterparty's customers"),
    Category("Competitive Restriction Exception", "Restrictive", "Exceptions or carve-outs to restrictive covenants"),
    Category("No-Solicit Of Employees", "Restrictive", "Restriction on soliciting or hiring employees"),
    Category("Non-Disparagement", "Restrictive", "Requirement not to disparage the counterparty"),
    Category("Termination For Convenience", "Termination", "A party can terminate without cause"),
    Category("Rofr/Rofo/Rofn", "Restrictive", "Right of first refusal, offer or negotiation"),
    Category("Change Of Control", "Termination", "Rights triggered by a change of control"),
    Category("Anti-Assignment", "Termination", "Consent or notice required to assign the contract"),
    Category("Revenue/Profit Sharing", "Financial", "Revenue or profit sharing with the counterparty"),
    Category("Price Restrictions", "Financial", "Restriction on raising or reducing prices"),
    Category("Minimum Commitment", "Financial", "Minimum order size or purchase amount"),
    Category("Volume Restriction", "Financial", "Fee increase or consent requirement above a volume threshold"),
    Category("Ip Ownership Assignment", "IP", "IP created by one party becomes property of the counterparty"),
    Category("Joint Ip Ownership", "IP", "Joint or shared ownership of intellectual property"),
    Category("License Grant", "IP", "A license granted by one party to its counterparty"),
    Category("Non-Transferable License", "IP", "Limit on transferring a licence to a third party"),
    Category("Affiliate License-Licensor", "IP", "Licence grant by affiliates of the licensor"),
    Category("Affiliate License-Licensee", "IP", "Licence grant to the licensee's affiliates"),
    Category("Unlimited/All-You-Can-Eat-License", "IP", "Unlimited usage licence"),
    Category("Irrevocable Or Perpetual License", "IP", "Irrevocable or perpetual licence"),
    Category("Source Code Escrow", "IP", "Source code must be deposited with an escrow agent"),
    Category("Post-Termination Services", "Termination", "Obligations that continue after termination"),
    Category("Audit Rights", "Financial", "Right to audit books, records or physical locations"),
    Category("Uncapped Liability", "Liability", "A party's liability is uncapped"),
    Category("Cap On Liability", "Liability", "A cap on liability"),
    Category("Liquidated Damages", "Liability", "Liquidated damages for breach or termination"),
    Category("Warranty Duration", "Liability", "Duration of a warranty"),
    Category("Insurance", "Liability", "Requirement to maintain insurance"),
    Category("Covenant Not To Sue", "Liability", "A party is restricted from contesting IP or bringing claims"),
    Category("Third Party Beneficiary", "General", "A non-party can enforce rights under the contract"),
]
CATEGORY_NAMES = [c.name for c in CATEGORIES]
CATEGORY_BY_NAME = {c.name: c for c in CATEGORIES}

# Metadata categories are better served by extraction than by a risk lens.
METADATA_CATEGORIES = {"Document Name", "Parties", "Agreement Date", "Effective Date",
                       "Expiration Date", "Governing Law"}
