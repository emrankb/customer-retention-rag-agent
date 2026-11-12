"""Customer records used by the retention assistant."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

CUSTOMER_COLUMNS = (
    "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Exited",
)
SENTIMENT_COLUMN = "Sentiment_Label"
SENTIMENT_VALUES = frozenset({"Positive", "Neutral", "Negative"})


@dataclass(frozen=True)
class Customer:
    customer_id: str
    surname: str
    credit_score: int
    geography: str
    gender: str
    age: int
    tenure: int
    balance: Decimal
    num_products: int
    has_credit_card: bool
    is_active_member: bool
    estimated_salary: Decimal
    exited: bool
    sentiment_label: str | None

    def as_context(self) -> dict[str, object]:
        """Use source values, without interpreting Exited as a risk prediction."""
        return {
            "customer_id": self.customer_id,
            "surname": self.surname,
            "credit_score": self.credit_score,
            "geography": self.geography,
            "gender": self.gender,
            "age": self.age,
            "tenure_years": self.tenure,
            "balance": str(self.balance),
            "num_products": self.num_products,
            "has_credit_card": self.has_credit_card,
            "is_active_member": self.is_active_member,
            "estimated_salary": str(self.estimated_salary),
            "historical_exited": self.exited,
            "sentiment_label": self.sentiment_label,
        }


def _required_text(value: str | None, field: str, line: int) -> str:
    result = (value or "").strip()
    if not result:
        raise ValueError(f"Row {line}: missing {field}.")
    return result


def _nonnegative_int(value: str | None, field: str, line: int) -> int:
    raw = _required_text(value, field, line)
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"Row {line}: {field} must be a nonnegative integer.")
    return int(raw)


def _binary(value: str | None, field: str, line: int) -> bool:
    raw = _required_text(value, field, line)
    if raw not in {"0", "1"}:
        raise ValueError(f"Row {line}: {field} must be 0 or 1.")
    return raw == "1"


def _amount(value: str | None, field: str, line: int) -> Decimal:
    raw = _required_text(value, field, line)
    try:
        amount = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"Row {line}: {field} is not numeric.") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"Row {line}: {field} must be a finite nonnegative value.")
    return amount


def _parse_record(row: dict, line: int, has_sentiment: bool) -> Customer:
    customer_id = _required_text(row.get("CustomerId"), "CustomerId", line)
    if not customer_id.isascii() or not customer_id.isdecimal():
        raise ValueError(f"Row {line}: CustomerId must contain only digits.")
    sentiment = None
    if has_sentiment:
        sentiment = _required_text(row.get(SENTIMENT_COLUMN), SENTIMENT_COLUMN, line)
        if sentiment not in SENTIMENT_VALUES:
            raise ValueError(f"Row {line}: unknown Sentiment_Label.")
    return Customer(
        customer_id=customer_id,
        surname=_required_text(row.get("Surname"), "Surname", line),
        credit_score=_nonnegative_int(row.get("CreditScore"), "CreditScore", line),
        geography=_required_text(row.get("Geography"), "Geography", line),
        gender=_required_text(row.get("Gender"), "Gender", line),
        age=_nonnegative_int(row.get("Age"), "Age", line),
        tenure=_nonnegative_int(row.get("Tenure"), "Tenure", line),
        balance=_amount(row.get("Balance"), "Balance", line),
        num_products=_nonnegative_int(row.get("NumOfProducts"), "NumOfProducts", line),
        has_credit_card=_binary(row.get("HasCrCard"), "HasCrCard", line),
        is_active_member=_binary(row.get("IsActiveMember"), "IsActiveMember", line),
        estimated_salary=_amount(row.get("EstimatedSalary"), "EstimatedSalary", line),
        exited=_binary(row.get("Exited"), "Exited", line),
        sentiment_label=sentiment,
    )


def load_customers(path: str | Path) -> dict[str, Customer]:
    """Read one supplied customer CSV and reject malformed or duplicate records."""
    source = Path(path)
    records: dict[str, Customer] = {}
    try:
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("Customer CSV is empty or has no header.")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Customer CSV contains duplicate column names.")
            missing = set(CUSTOMER_COLUMNS).difference(reader.fieldnames)
            if missing:
                raise ValueError("Customer CSV is missing: " + ", ".join(sorted(missing)))
            has_sentiment = SENTIMENT_COLUMN in reader.fieldnames
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"Row {row_number}: wrong number of CSV fields.")
                customer = _parse_record(row, row_number, has_sentiment)
                if customer.customer_id in records:
                    raise ValueError(f"Row {row_number}: duplicate CustomerId.")
                records[customer.customer_id] = customer
    except UnicodeError as exc:
        raise ValueError(f"Customer CSV is not valid UTF-8: {source.name}.") from exc
    if not records:
        raise ValueError("Customer CSV has no customer records.")
    return records


class CustomerDB:
    """Indexed customer lookup. No new churn prediction is calculated here."""

    def __init__(self, csv_path: str | Path):
        self._records = load_customers(csv_path)

    def __len__(self) -> int:
        return len(self._records)

    def get_context(self, customer_id: str | int) -> dict[str, object] | None:
        if isinstance(customer_id, bool) or not isinstance(customer_id, (str, int)):
            raise ValueError("Customer ID must be a string of digits or an integer.")
        identifier = str(customer_id).strip()
        if not identifier.isascii() or not identifier.isdecimal():
            raise ValueError("Customer ID must contain only digits.")
        customer = self._records.get(identifier)
        return customer.as_context() if customer else None
