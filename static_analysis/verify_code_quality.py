
from regex import P


MINIMUM_RATE = 9.8


def test_code_quality():
    with open("static_analysis/report_lynter/report.txt", "r") as file:

        file_lines = file.readlines()[-10:]

        for line in file_lines:
            if "rated at" in line:
                rate = line.split("rated at")[1].strip().split("/")[0]

                rate = float(rate)
                assert (rate > MINIMUM_RATE), f"Your code quality is, {rate} which is below the minimum {MINIMUM_RATE}"


test_code_quality()
