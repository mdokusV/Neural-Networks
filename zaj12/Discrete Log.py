from math import gcd
from random import randint

MAX: int = 37 * 71

while True:
    guess = randint(1, MAX)
    print(f"{guess=}")

    new_gcd = gcd(guess, MAX)
    if new_gcd != 1:
        print(f"Divisor: {new_gcd}")
        exit(0)

    print(f"{new_gcd=}")

    r = 1
    expo = guess
    while pow(guess, r, MAX) != 1:
        r += 1
    print(f"{r=}")

    if r % 2 != 0:
        print("Wrong guess", end="\n\n")
        continue

    guess_left = gcd(pow(guess, r // 2, MAX) - 1, MAX)
    guess_right = gcd(pow(guess, r // 2, MAX) + 1, MAX)

    if guess_left != 1:
        print(f"Divisor left: {guess_left}")

    if guess_right != 1:
        print(f"Divisor right: {guess_right}")

    if guess_left != 1 and guess_right != 1:
        exit(0)

    print("No divisor found", end="\n\n")
