from math import gcd, log
from random import randint
from sympy import totient, divisors

MAX: int = 899959983751 * 899959984169
print(len(str(MAX)))
print(f"{MAX=}")

phi_MAX = totient(MAX)
divs_phi_MAX = divisors(phi_MAX)

while True:
    guess = randint(1, MAX)
    print(f"{guess=}")

    new_gcd = gcd(guess, MAX)
    if new_gcd != 1:
        print(f"Divisor: {new_gcd}")
        exit(0)

    print(f"{new_gcd=}")

    exponent = 1
    for d in divs_phi_MAX:
        if pow(guess, d, MAX) == 1:
            exponent = d
            break
    print(f"{exponent=}")

    if exponent % 2 != 0:
        print("Wrong guess", end="\n\n")
        continue

    guess_left = gcd(pow(guess, exponent // 2, MAX) - 1, MAX)
    guess_right = gcd(pow(guess, exponent // 2, MAX) + 1, MAX)

    if guess_left != 1:
        print(f"Divisor left: {guess_left}")

    if guess_right != 1:
        print(f"Divisor right: {guess_right}")

    if guess_left != 1 and guess_right != 1:
        exit(0)

    print("No divisor found", end="\n\n")
