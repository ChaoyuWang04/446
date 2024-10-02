import numpy as np
from typing import Union


class Polynomial:
    def __init__(self, coefficients: Union[list, np.ndarray]):
        # Remove leading zeros
        while len(coefficients) > 1 and coefficients[0] == 0:
            coefficients = coefficients[1:]
        self.coefficients = np.array(coefficients, dtype=int)
        self.order = len(self.coefficients) - 1

    @staticmethod
    def from_string(s: str) -> 'Polynomial':
        s = s.replace(" ", "").replace("-", "+-")
        if s[0] == "+":
            s = s[1:]
        terms = s.split("+")
        coeff_dict = {}
        for term in terms:
            if term:
                if 'x^' in term:
                    coeff, power = term.split('x^')
                elif 'x' in term:
                    coeff, power = term.split('x')
                    power = '1'
                else:
                    coeff, power = term, '0'

                coeff = coeff.replace('*', '')  # Remove asterisks
                if coeff == '' or coeff == '+':
                    coeff = 1
                elif coeff == '-':
                    coeff = -1
                else:
                    coeff = int(coeff)

                power = int(power)
                coeff_dict[power] = coeff_dict.get(power, 0) + coeff

        if coeff_dict:
            order = max(coeff_dict.keys())
            coeffs = [coeff_dict.get(i, 0) for i in range(order + 1)]
            return Polynomial(coeffs[::-1])
        else:
            return Polynomial([0])

    def __repr__(self):
        terms = []
        for i, coeff in enumerate(self.coefficients[::-1]):
            if coeff != 0:
                if i == 0:
                    terms.append(str(coeff))
                elif i == 1:
                    terms.append(f"{coeff}*x" if abs(coeff) != 1 else ("-x" if coeff == -1 else "x"))
                else:
                    terms.append(f"{coeff}*x^{i}" if abs(coeff) != 1 else (f"-x^{i}" if coeff == -1 else f"x^{i}"))
        return " + ".join(terms).replace("+ -", "- ") or "0"

    def __add__(self, other):
        max_order = max(self.order, other.order)
        new_coeffs = np.zeros(max_order + 1, dtype=int)
        new_coeffs[-len(self.coefficients):] += self.coefficients
        new_coeffs[-len(other.coefficients):] += other.coefficients
        return Polynomial(new_coeffs)

    def __sub__(self, other):
        return self + Polynomial([-c for c in other.coefficients])

    def __mul__(self, other):
        new_order = self.order + other.order
        new_coeffs = np.zeros(new_order + 1, dtype=int)
        for i, c1 in enumerate(self.coefficients):
            for j, c2 in enumerate(other.coefficients):
                new_coeffs[i + j] += c1 * c2
        return Polynomial(new_coeffs)

    def __eq__(self, other):
        return np.array_equal(self.coefficients, other.coefficients)

    def __truediv__(self, other):
        return RationalPolynomial(self, other)


class RationalPolynomial:
    def __init__(self, numerator: Polynomial, denominator: Polynomial):
        self.numerator = numerator
        self.denominator = denominator
        self._simplify()

    @staticmethod
    def from_string(s: str) -> 'RationalPolynomial':
        num_str, denom_str = s.split('/')
        numerator = Polynomial.from_string(num_str.strip('()'))
        denominator = Polynomial.from_string(denom_str.strip('()'))
        return RationalPolynomial(numerator, denominator)

    def _simplify(self):
        # A placeholder
        pass

    def __repr__(self):
        return f"({self.numerator}) / ({self.denominator})"

    def __add__(self, other):
        new_num = self.numerator * other.denominator + other.numerator * self.denominator
        new_denom = self.denominator * other.denominator
        return RationalPolynomial(new_num, new_denom)

    def __sub__(self, other):
        new_num = self.numerator * other.denominator - other.numerator * self.denominator
        new_denom = self.denominator * other.denominator
        return RationalPolynomial(new_num, new_denom)

    def __mul__(self, other):
        new_num = self.numerator * other.numerator
        new_denom = self.denominator * other.denominator
        return RationalPolynomial(new_num, new_denom)

    def __truediv__(self, other):
        new_num = self.numerator * other.denominator
        new_denom = self.denominator * other.numerator
        return RationalPolynomial(new_num, new_denom)

    def __eq__(self, other):
        return self.numerator * other.denominator == other.numerator * self.denominator

