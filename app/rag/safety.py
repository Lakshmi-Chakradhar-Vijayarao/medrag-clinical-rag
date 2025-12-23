# app/rag/safety.py
from typing import List


def check_scope_warnings(query: str) -> List[str]:
    """
    Very simple, rule-based safety layer.
    Flags queries that are out-of-scope for this assistant:
    - Dosing / mg / titration
    - Pregnancy / lactation
    - Paediatrics
    - Emergencies / acute situations
    """
    q = query.lower()
    warnings: List[str] = []

    # Dosing / prescribing
    dose_keywords = [
        "dose",
        "dosing",
        "mg",
        "milligram",
        "milligrams",
        "units",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "titrate",
        "titration",
        "how much",
        "how many",
    ]
    if any(k in q for k in dose_keywords):
        warnings.append(
            "Scope warning: This assistant does NOT provide dosing or prescribing "
            "recommendations. Always consult local guidelines and a licensed clinician for "
            "dose decisions."
        )

    # Pregnancy / lactation
    pregnancy_keywords = [
        "pregnant",
        "pregnancy",
        "lactation",
        "breastfeeding",
        "breast-feeding",
        "trimester",
    ]
    if any(k in q for k in pregnancy_keywords):
        warnings.append(
            "Scope warning: This assistant is not validated for pregnancy or "
            "lactation-specific recommendations. Refer to specialist guidelines."
        )

    # Paediatrics
    paeds_keywords = [
        "child",
        "children",
        "paediatric",
        "pediatric",
        "neonate",
        "infant",
        "years old",
        "year-old",
        "yr old",
    ]
    if any(k in q for k in paeds_keywords):
        warnings.append(
            "Scope warning: This assistant is not intended for paediatric patients. "
            "Consult paediatric-specific resources."
        )

    # Emergencies / acute
    emergency_keywords = [
        "emergency",
        "stat",
        "resuscitate",
        "resuscitation",
        "shock",
        "collapse",
        "cardiac arrest",
        "unresponsive",
        "acute chest pain",
        "severe shortness of breath",
    ]
    if any(k in q for k in emergency_keywords):
        warnings.append(
            "Scope warning: This assistant is NOT for emergency decision-making. "
            "In emergencies, follow local protocols and on-site medical leadership."
        )

    return warnings
