def calculate_tax(amount: float, rate: float) -> float:
    """Return the tax owed on `amount` at `rate`."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    return round(amount * rate, 2)
