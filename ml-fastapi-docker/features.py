"""Human-readable feature schema for the Statlog German Credit dataset.

The trained model still expects the original UCI codes (``A11``, ``A12``, ...).
This module is the single source of truth used by both the API and the
Streamlit UI: it maps friendly keys/labels to the codes the model understands.

Reference:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Category:
    """One possible value of a categorical feature."""

    key: str
    code: str
    label: str
    hint: str = ""
    short: str = ""


@dataclass(frozen=True)
class Feature:
    """Description of a single model input."""

    name: str
    label: str
    group: str
    kind: str  # "categorical" | "numeric"
    help: str
    categories: tuple[Category, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    step: int = 1
    unit: str = ""
    value_labels: dict[int, str] = field(default_factory=dict)


FEATURES: tuple[Feature, ...] = (
    Feature(
        name="checking_account",
        label="Checking account status",
        group="Finances",
        kind="categorical",
        help="Balance on the applicant's checking account, in Deutsche Mark.",
        categories=(
            Category("overdrawn", "A11", "Overdrawn (below 0 DM)", "The account is in the red.", short="Overdrawn"),
            Category("low_balance", "A12", "Low balance (0 – 200 DM)", short="Low balance"),
            Category(
                "healthy_balance",
                "A13",
                "Healthy balance (200 DM or more)",
                "200 DM or more, or salary payments for at least one year.",
                short="Healthy balance",
            ),
            Category("no_account", "A14", "No checking account"),
        ),
    ),
    Feature(
        name="duration_months",
        label="Loan duration",
        group="Loan",
        kind="numeric",
        help="How long the requested credit will run.",
        minimum=4,
        maximum=72,
        unit="months",
    ),
    Feature(
        name="credit_history",
        label="Credit history",
        group="Finances",
        kind="categorical",
        help="How reliably the applicant has repaid credits so far.",
        categories=(
            Category("no_credits", "A30", "No credits taken / all repaid"),
            Category("all_repaid_here", "A31", "All credits at this bank repaid"),
            Category("existing_repaid", "A32", "Existing credits paid back so far"),
            Category("past_delays", "A33", "Past payment delays"),
            Category("critical_account", "A34", "Critical account / credits elsewhere", short="Critical account"),
        ),
    ),
    Feature(
        name="purpose",
        label="Loan purpose",
        group="Loan",
        kind="categorical",
        help="What the money will be used for.",
        categories=(
            Category("new_car", "A40", "New car"),
            Category("used_car", "A41", "Used car"),
            Category("furniture", "A42", "Furniture / equipment"),
            Category("radio_tv", "A43", "Radio / television"),
            Category("appliances", "A44", "Domestic appliances"),
            Category("repairs", "A45", "Repairs"),
            Category("education", "A46", "Education"),
            Category("retraining", "A48", "Retraining"),
            Category("business", "A49", "Business"),
            Category("other", "A410", "Other"),
        ),
    ),
    Feature(
        name="credit_amount",
        label="Credit amount",
        group="Loan",
        kind="numeric",
        help="Size of the requested credit.",
        minimum=250,
        maximum=20000,
        step=100,
        unit="DM",
    ),
    Feature(
        name="savings_account",
        label="Savings account / bonds",
        group="Finances",
        kind="categorical",
        help="Savings the applicant has with the bank.",
        categories=(
            Category("below_100", "A61", "Under 100 DM"),
            Category("100_to_500", "A62", "100 – 500 DM"),
            Category("500_to_1000", "A63", "500 – 1,000 DM"),
            Category("above_1000", "A64", "1,000 DM or more"),
            Category("unknown", "A65", "Unknown / no savings account"),
        ),
    ),
    Feature(
        name="employment_since",
        label="Present employment since",
        group="Applicant",
        kind="categorical",
        help="How long the applicant has been in the current job.",
        categories=(
            Category("unemployed", "A71", "Unemployed"),
            Category("under_1y", "A72", "Less than 1 year"),
            Category("1_to_4y", "A73", "1 – 4 years"),
            Category("4_to_7y", "A74", "4 – 7 years"),
            Category("over_7y", "A75", "7 years or more"),
        ),
    ),
    Feature(
        name="installment_rate",
        label="Installment rate",
        group="Finances",
        kind="numeric",
        help="Share of disposable income the installment takes.",
        minimum=1,
        maximum=4,
        value_labels={
            1: "1 — lowest (up to 10%)",
            2: "2 — low (10 – 20%)",
            3: "3 — high (20 – 30%)",
            4: "4 — highest (over 30%)",
        },
    ),
    Feature(
        name="personal_status_sex",
        label="Personal status & sex",
        group="Applicant",
        kind="categorical",
        help="Marital status and sex as recorded in the dataset.",
        categories=(
            Category("male_divorced", "A91", "Male, divorced / separated"),
            Category("female_divorced_married", "A92", "Female, divorced / separated / married", short="Female, divorced/married"),
            Category("male_single", "A93", "Male, single"),
            Category("male_married", "A94", "Male, married / widowed"),
        ),
    ),
    Feature(
        name="other_debtors",
        label="Other debtors / guarantors",
        group="Obligations",
        kind="categorical",
        help="Anyone else liable for this credit.",
        categories=(
            Category("none", "A101", "None"),
            Category("co_applicant", "A102", "Co-applicant"),
            Category("guarantor", "A103", "Guarantor"),
        ),
    ),
    Feature(
        name="residence_since",
        label="Present residence since",
        group="Obligations",
        kind="numeric",
        help="Years at the current address (the dataset caps this at 4).",
        minimum=1,
        maximum=4,
        value_labels={1: "1 year", 2: "2 years", 3: "3 years", 4: "4 years"},
    ),
    Feature(
        name="property",
        label="Property",
        group="Obligations",
        kind="categorical",
        help="Property the applicant owns.",
        categories=(
            Category("real_estate", "A121", "Real estate"),
            Category(
                "building_society",
                "A122",
                "Building society savings / life insurance",
                short="Building society",
            ),
            Category("car_or_other", "A123", "Car or other (not savings)"),
            Category("none", "A124", "Unknown / no property"),
        ),
    ),
    Feature(
        name="age",
        label="Age",
        group="Applicant",
        kind="numeric",
        help="Applicant's age in years.",
        minimum=18,
        maximum=75,
        unit="years",
    ),
    Feature(
        name="other_installment_plans",
        label="Other installment plans",
        group="Obligations",
        kind="categorical",
        help="Other ongoing installment agreements.",
        categories=(
            Category("bank", "A141", "Bank"),
            Category("stores", "A142", "Stores"),
            Category("none", "A143", "None"),
        ),
    ),
    Feature(
        name="housing",
        label="Housing",
        group="Obligations",
        kind="categorical",
        help="Housing situation of the applicant.",
        categories=(
            Category("rent", "A151", "Rent"),
            Category("own", "A152", "Own"),
            Category("free", "A153", "Free of charge"),
        ),
    ),
    Feature(
        name="existing_credits",
        label="Existing credits at this bank",
        group="Finances",
        kind="numeric",
        help="Number of credits the applicant already runs at this bank.",
        minimum=1,
        maximum=4,
        value_labels={1: "1 credit", 2: "2 credits", 3: "3 credits", 4: "4 or more"},
    ),
    Feature(
        name="job",
        label="Job",
        group="Applicant",
        kind="categorical",
        help="Qualification level of the applicant's job.",
        categories=(
            Category("unemployed_unskilled", "A171", "Unemployed / unskilled (non-resident)", short="Unemployed / unskilled"),
            Category("unskilled_resident", "A172", "Unskilled (resident)"),
            Category("skilled", "A173", "Skilled employee / official"),
            Category(
                "management",
                "A174",
                "Management / self-employed / highly qualified",
                short="Management",
            ),
        ),
    ),
    Feature(
        name="maintenance_for",
        label="People to support",
        group="Obligations",
        kind="numeric",
        help="Number of people the applicant provides maintenance for.",
        minimum=1,
        maximum=2,
        value_labels={1: "1 person", 2: "2 people"},
    ),
    Feature(
        name="telephone",
        label="Telephone",
        group="Applicant",
        kind="categorical",
        help="Whether the applicant has a phone registered in their name.",
        categories=(
            Category("none", "A191", "None"),
            Category("registered", "A192", "Yes, registered under customer's name", short="Yes, registered"),
        ),
    ),
    Feature(
        name="foreign_worker",
        label="Foreign worker",
        group="Applicant",
        kind="categorical",
        help="Whether the applicant is a foreign worker.",
        categories=(
            Category("yes", "A201", "Yes"),
            Category("no", "A202", "No"),
        ),
    ),
)

FEATURES_BY_NAME: dict[str, Feature] = {feature.name: feature for feature in FEATURES}

COLUMN_NAMES: tuple[str, ...] = tuple(feature.name for feature in FEATURES) + ("target",)

CATEGORICAL_NAMES: tuple[str, ...] = tuple(
    feature.name for feature in FEATURES if feature.kind == "categorical"
)

NUMERIC_NAMES: tuple[str, ...] = tuple(
    feature.name for feature in FEATURES if feature.kind == "numeric"
)

GROUP_ORDER: tuple[str, ...] = ("Applicant", "Loan", "Finances", "Obligations")

#: A typical, moderately safe applicant used to prefill the form.
DEFAULT_INPUT: dict[str, Any] = {
    "checking_account": "healthy_balance",
    "duration_months": 24,
    "credit_history": "existing_repaid",
    "purpose": "furniture",
    "credit_amount": 3000,
    "savings_account": "100_to_500",
    "employment_since": "1_to_4y",
    "installment_rate": 2,
    "personal_status_sex": "male_single",
    "other_debtors": "none",
    "residence_since": 2,
    "property": "car_or_other",
    "age": 30,
    "other_installment_plans": "none",
    "housing": "rent",
    "existing_credits": 1,
    "job": "skilled",
    "maintenance_for": 1,
    "telephone": "registered",
    "foreign_worker": "no",
}


def _normalize(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _find_category(feature: Feature, value: Any) -> Category:
    needle = _normalize(value)
    for category in feature.categories:
        if needle in (
            _normalize(category.key),
            _normalize(category.code),
            _normalize(category.label),
        ):
            return category
    allowed = ", ".join(category.key for category in feature.categories)
    raise ValueError(
        f"Unknown value {value!r} for '{feature.name}'. Allowed values: {allowed}"
    )


def resolve(name: str, value: Any) -> str:
    """Return the UCI code (e.g. ``A11``) for a key, code or human label."""

    feature = FEATURES_BY_NAME[name]
    if feature.kind != "categorical":
        raise ValueError(f"'{name}' is numeric and has no category codes")
    return _find_category(feature, value).code


def category(name: str, value: Any) -> Category:
    """Return the full category record for a key, code or human label."""

    return _find_category(FEATURES_BY_NAME[name], value)


def display_value(name: str, value: Any) -> str:
    """Human-readable representation of any feature value."""

    feature = FEATURES_BY_NAME[name]
    if feature.kind == "categorical":
        return _find_category(feature, value).label
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    if number in feature.value_labels:
        return feature.value_labels[number]
    unit = f" {feature.unit}" if feature.unit else ""
    return f"{number:,}{unit}"


def compact_value(name: str, value: Any) -> str:
    """Shorter representation for tight spaces such as the sidebar."""

    feature = FEATURES_BY_NAME[name]
    if feature.kind == "categorical":
        found = _find_category(feature, value)
        return found.short or found.label
    return display_value(name, value)


def to_model_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a human-friendly payload into the raw row expected by the model."""

    row: dict[str, Any] = {}
    for feature in FEATURES:
        value = payload[feature.name]
        if feature.kind == "categorical":
            row[feature.name] = resolve(feature.name, value)
        else:
            row[feature.name] = int(value)
    return row


def schema() -> list[dict[str, Any]]:
    """Serializable description of every input, used by the API docs."""

    result: list[dict[str, Any]] = []
    for feature in FEATURES:
        item: dict[str, Any] = {
            "name": feature.name,
            "label": feature.label,
            "group": feature.group,
            "kind": feature.kind,
            "help": feature.help,
        }
        if feature.kind == "categorical":
            item["options"] = [
                {"key": c.key, "code": c.code, "label": c.label, "short": c.short or c.label, "hint": c.hint}
                for c in feature.categories
            ]
        else:
            item.update(
                {
                    "minimum": feature.minimum,
                    "maximum": feature.maximum,
                    "step": feature.step,
                    "unit": feature.unit,
                    "value_labels": feature.value_labels,
                }
            )
        result.append(item)
    return result
