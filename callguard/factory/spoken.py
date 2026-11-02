"""Write numbers the way streaming ASR (Nemotron) emits them: "$499" -> "four hundred ninety nine dollars",
"555-0123" -> "five five five oh one two three", "482916" -> "four eight two nine one six"."""
import re

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
DIG = "zero one two three four five six seven eight nine".split()


def words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))
    if n < 1_000_000:
        return words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + words(n % 1000))
    return " ".join(DIG[int(c)] for c in str(n))


def digits(s, oh=False):
    return " ".join(("oh" if oh and c == "0" else DIG[int(c)]) for c in s if c.isdigit())


PHONE = re.compile(r"\b(?:\+?1[-. ])?\(?(\d{3})\)?[-. ](\d{3})[-. ](\d{4})\b|\b(\d{3})[-. ](\d{4})\b")
MONEY = re.compile(r"\$(\d[\d,]*)(?:\.(\d{2}))?")
LONG = re.compile(r"\b\d{4,}\b")
SMALL = re.compile(r"(?<![\w.])\d{1,3}(?!\w|\.\d)")


def spoken(text):
    def phone(m):
        return digits("".join(g for g in m.groups() if g), oh=True)

    def money(m):
        n = int(m.group(1).replace(",", ""))
        out = words(n) + (" dollar" if n == 1 else " dollars")
        if m.group(2) and int(m.group(2)):
            out += " and " + words(int(m.group(2))) + " cents"
        return out

    text = PHONE.sub(phone, text)
    text = MONEY.sub(money, text)
    text = LONG.sub(lambda m: digits(m.group(0)), text)
    text = SMALL.sub(lambda m: words(int(m.group(0))), text)
    return text


def spoken_row(row):
    """Apply to every transcript turn and to label strings, so evidence quotes stay exact substrings."""
    for t in row["turns"]:
        t["text"] = spoken(t["text"])
        L = t["label"]
        for k in ("evidence", "coach_line", "caretaker_note"):
            if L.get(k):
                L[k] = spoken(L[k])
    row["spoken_numbers"] = True
    return row


if __name__ == "__main__":
    for s in ["Your card ending 4821 was charged $499.00 at 3 PM.", "Call me at 555-0123 or 555-013-4567.",
              "Read me the code 482916.", "I'm 10 and my mom gets home at 6."]:
        print(s, "->", spoken(s))
