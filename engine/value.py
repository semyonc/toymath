import re
import math
from notation import Symbol, Notation


class Value(object):
    """Value"""

    def get_int(self):
        return None

    def get_float(self):
        return None

    def get_frac(self):
        return None

    def add(self, other):
        return None

    def sub(self, other):
        return None

    def mul(self, other):
        return None

    def div(self, other):
        return None

    def power(self, n):
        return None

    def abs(self):
        return self

    @staticmethod
    def create(x):
        if isinstance(x, int):
            return IntegerValue(x)
        elif isinstance(x, float):
            return FloatValue(x)
        elif isinstance(x, Value):
            return x
        else:
            return None


class IntegerValue(Value):
    """IntegerValue"""

    typeKey = 1

    def __init__(self, val):
        self.val = val

    def get_int(self):
        return self

    def get_float(self):
        return FloatValue(float(self.val))

    def get_frac(self):
        return FracValue(self.val, 1)

    def __eq__(self, o):
        return isinstance(o, IntegerValue) and o.val == self.val

    def __ne__(self, o):
        return isinstance(o, IntegerValue) and o.val != self.val

    def __gt__(self, o):
        return isinstance(o, IntegerValue) and self.val > o.val

    def __ge__(self, o):
        return isinstance(o, IntegerValue) and self.val >= o.val

    def __lt__(self, o):
        return isinstance(o, IntegerValue) and self.val < o.val

    def __le__(self, o):
        return isinstance(o, IntegerValue) and self.val <= o.val

    def __hash__(self):
        return self.val.__hash__()

    def __repr__(self):
        if self.val < 0:
            return '{' + str(-self.val) + '}'
        else:
            return '{' + str(self.val) + '}'

    def add(self, other):
        assert isinstance(other, IntegerValue)
        return IntegerValue(self.val + other.val)

    def sub(self, other):
        assert isinstance(other, IntegerValue)
        return IntegerValue(self.val - other.val)

    def mul(self, other):
        assert isinstance(other, IntegerValue)
        return IntegerValue(self.val * other.val)

    def div(self, other):
        assert isinstance(other, IntegerValue)
        return IntegerValue(int(self.val / other.val))

    def power(self, n):
        assert isinstance(n, IntegerValue)
        if n.val >= 0:
            return IntegerValue(int(self.val ** n.val))
        else:
            return FracValue(1, int(self.val ** (-n.val)))

    def abs(self):
        if self.val < 0:
            return IntegerValue(-self.val)
        return self


class FloatValue(Value):
    """FloatValue"""

    typeKey = 2

    def __init__(self, val):
        self.val = val

    def get_int(self):
        return IntegerValue(int(self.val))

    def get_float(self):
        return self

    re_digit = re.compile(r'[0-9]+(\.[0-9]+)?')

    def get_frac(self):
        if FloatValue.re_digit.match(self.val):
            val = str(self.val)
            parts = val.split('.')
            res = FracValue(int(parts[0]), 1)
            if len(parts) > 1:
                num = int(parts[1])
                denom = 10 ** len(parts[1])
                res = res.add(FracValue(num, denom))
            return res
        return None

    def __eq__(self, o):
        return isinstance(o, FloatValue) and o.val == self.val

    def __ne__(self, o):
        return isinstance(o, FloatValue) and o.val != self.val

    def __gt__(self, o):
        return isinstance(o, FloatValue) and self.val > o.val

    def __ge__(self, o):
        return isinstance(o, FloatValue) and self.val >= o.val

    def __lt__(self, o):
        return isinstance(o, FloatValue) and self.val < o.val

    def __le__(self, o):
        return isinstance(o, FloatValue) and self.val <= o.val

    def __hash__(self):
        return self.val.__hash__()

    def __repr__(self):
        return '{' + str(math.fabs(self.val)) + '}'

    def add(self, other):
        assert isinstance(other, FloatValue)
        return FloatValue(self.val + other.val)

    def sub(self, other):
        assert isinstance(other, FloatValue)
        return FloatValue(self.val - other.val)

    def mul(self, other):
        assert isinstance(other, FloatValue)
        return FloatValue(self.val * other.val)

    def div(self, other):
        assert isinstance(other, FloatValue)
        return FloatValue(self.val / other.val)

    def abs(self):
        if self.val < 0:
            return FloatValue(math.fabs(self.val))
        return self


class FracValue(Value):
    """FracValue"""

    typeKey = 3

    def __init__(self, num, denom):
        gcd = math.gcd(num, denom)
        self.num = int(num / gcd)
        self.denom = int(denom / gcd)

    def get_int(self):
        return IntegerValue(int(math.trunc(self.num / self.denom)))

    def get_float(self):
        return FloatValue(self.num / self.denom)

    def get_frac(self):
        return self

    def __repr__(self):
        num = self.num
        denom = self.denom
        if num < 0:
            num = -num
        integer = int(math.trunc(num / denom))
        rem = num - integer * denom
        if rem > 0:
            result = f'\\frac{{{rem}}}{{{denom}}}'
            if integer > 0:
                result = f'{{{integer}{result}}}'
        else:
            result = f'{{{integer}}}'
        return result

    def __eq__(self, o):
        return isinstance(o, FracValue) and o.num == self.num and o.denom == self.denom

    def __ne__(self, o):
        return isinstance(o, FracValue) and not (o.num == self.num and o.denom == self.denom)

    def __gt__(self, o):
        return isinstance(o, FracValue) and o.num * self.denom < self.num * o.denom

    def __ge__(self, o):
        return isinstance(o, FracValue) and o.num * self.denom <= self.num * o.denom

    def __lt__(self, o):
        return isinstance(o, FracValue) and o.num * self.denom > self.num * o.denom

    def __le__(self, o):
        return isinstance(o, FracValue) and o.num * self.denom >= self.num * o.denom

    def __hash__(self):
        return hash((self.num, self.denom))

    def add(self, other):
        assert isinstance(other, FracValue)
        num = self.num * other.denom + other.num * self.denom
        denom = self.denom * other.denom
        return FracValue(num, denom)

    def sub(self, other):
        assert isinstance(other, FracValue)
        num = self.num * other.denom - other.num * self.denom
        denom = self.denom * other.denom
        return FracValue(num, denom)

    def mul(self, other):
        assert isinstance(other, FracValue)
        num = self.num * other.num
        denom = self.denom * other.denom
        return FracValue(num, denom)

    def div(self, other):
        assert isinstance(other, FracValue)
        if other.num < 0:
            num = self.num * (- other.denom)
            denom = self.denom * (- other.num)
        else:
            num = self.num * other.denom
            denom = self.denom * other.num
        return FracValue(num, denom)

    def power(self, n):
        assert isinstance(n, IntegerValue)
        if n.val == 0:
            return IntegerValue(1)
        elif n.val > 0:
            return FracValue(int(self.num ** n.val), int(self.denom ** n.val))
        else:
            return FracValue(int(self.denom ** (-n.val)), int(self.num ** (-n.val)))

    def abs(self):
        if self.num < 0:
            return FracValue(-self.num, self.denom)
        return self


def _promote(a, other):
    if type(a) != type(other):
        if isinstance(other, FracValue):
            return a.get_frac()
        if isinstance(other, FloatValue):
            return a.get_float()
    return a


def addition(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        x = _promote(x, y)
        y = _promote(y, x)
        return x.add(y)
    return None


def subtraction(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        x = _promote(x, y)
        y = _promote(y, x)
        return x.sub(y)
    return None


def multiplication(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        x = _promote(x, y)
        y = _promote(y, x)
        return x.mul(y)
    return None


def division(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        x = _promote(x, y)
        y = _promote(y, x)
        return x.div(y)
    return None


def power(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        return x.power(y)
    return None


def value_type(val):
    if isinstance(val, IntegerValue):
        return IntegerValue.typeKey
    elif isinstance(val, FloatValue):
        return FloatValue.typeKey
    elif isinstance(val, FracValue):
        return FracValue.typeKey
    else:
        return None


def equal_value(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        x = _promote(x, y)
        y = _promote(y, x)
        return x == y
    return False


def less_value(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        x = _promote(x, y)
        y = _promote(y, x)
        return x < y
    return False


def less_or_equal_value(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        x = _promote(x, y)
        y = _promote(y, x)
        return x <= y
    return False


def not_equal_value(x, y):
    x = Value.create(x)
    y = Value.create(y)
    if x is not None and y is not None:
        x = _promote(x, y)
        y = _promote(y, x)
        return x != y
    return False
