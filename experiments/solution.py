from decimal import Decimal, getcontext, ROUND_UP
from pathlib import Path

def round_up(x: float, dec_places: int) -> float | int:
    """
    Rounds a floating-point number up to a specified number of decimal places.
    
    Args:
        x (float): The number to be rounded up.
        dec_places (int): The number of decimal places to round up to.
    
    Returns:
        float | int: The rounded value as a float or int.
    """
    d = Decimal(str(x))
    getcontext().rounding = ROUND_UP
    quantized = d.quantize(Decimal('1.' + '0' * dec_places))
    if quantized == quantized.to_integral_value():
        return int(quantized)
    else:
        return float(quantized)
