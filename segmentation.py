HIGH = "highRisk"
MEDIUM = "mediumRisk"
LOW = "lowRisk"


def get_segment(contract, tenure, monthly_charges, threshold):
    if (contract == "Month-to-month") and (tenure < 12) and (monthly_charges > threshold):
        return HIGH

    elif ((contract == "One-year") and (tenure >= 12)) or \
            ((contract == "Month-to-month") and (tenure >= 12)):
        return MEDIUM

    return LOW


def get_action(segment):
    if segment == HIGH:
        return (
            "Özel indirim ve kontrat yenileme kampanyası",
            "Kısa tenure + yüksek ücret → churn riski yüksek"
        )

    elif segment == MEDIUM:
        return (
            "Sadakat programı ve küçük teşvikler",
            "Orta seviyede risk → müşteri elde tutulmalı"
        )

    return (
        "Teşekkür e-postası ve çapraz satış fırsatları",
        "Düşük risk → müşteri memnuniyeti artırılmalı"
    )


def predict(contract, tenure, monthly_charges, threshold):
    segment = get_segment(contract, tenure, monthly_charges, threshold)
    action, reason = get_action(segment)
    return segment, action, reason