#!/usr/bin/env python3
"""Generate independent fixed vectors for the project random protocol."""

from __future__ import print_function

import argparse
import math


MASK64 = (1 << 64) - 1
SPLIT_DOMAIN = 0x53504C49545F5631
SHUFFLE_DOMAIN = 0x53485546464C4531
TRANSLATION_X_DOMAIN = 0x5452414E535F5831
TRANSLATION_Y_DOMAIN = 0x5452414E535F5931
INITIALIZATION_DOMAIN = 0x494E49545F563031


def mix64(value):
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def derive(seed, domain, a, b):
    value = mix64((seed ^ domain) & MASK64)
    value = mix64((value ^ a) & MASK64)
    return mix64((value ^ b) & MASK64)


class SplitMix64(object):
    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        return mix64(self.state)

    def bounded(self, bound):
        if bound <= 0 or bound > MASK64:
            raise ValueError("bound must be in [1, 2^64 - 1]")
        threshold = ((-bound) & MASK64) % bound
        while True:
            value = self.next()
            if value >= threshold:
                return value % bound


def shuffled(values, seed):
    result = list(values)
    random = SplitMix64(seed)
    for index in range(len(result) - 1, 0, -1):
        other = random.bounded(index + 1)
        result[index], result[other] = result[other], result[index]
    return result


def split_indices(sample_count, validation_count, seed):
    stream = derive(seed, SPLIT_DOMAIN, sample_count, validation_count)
    permutation = shuffled(range(sample_count), stream)
    training_count = sample_count - validation_count
    return permutation[:training_count], permutation[training_count:]


def epoch_indices(canonical, seed, one_based_epoch):
    stream = derive(seed, SHUFFLE_DOMAIN, one_based_epoch, len(canonical))
    return shuffled(canonical, stream)


def translation(seed, one_based_epoch, original_index):
    x_seed = derive(seed, TRANSLATION_X_DOMAIN, one_based_epoch,
                    original_index)
    y_seed = derive(seed, TRANSLATION_Y_DOMAIN, one_based_epoch,
                    original_index)
    return (SplitMix64(x_seed).bounded(5) - 2,
            SplitMix64(y_seed).bounded(5) - 2)


def open_uniform(raw):
    # Center a 52-bit bin. Both endpoints remain exactly representable and
    # strictly inside (0, 1), including raw values 0 and 2^64 - 1.
    return ((raw >> 12) + 0.5) * (2.0 ** -52)


def initialized_prefix(seed, ordinal, fan_in, count):
    stream = derive(seed, INITIALIZATION_DOMAIN, ordinal, fan_in)
    random = SplitMix64(stream)
    standard_deviation = math.sqrt(2.0 / fan_in)
    values = []
    while len(values) < count:
        first = open_uniform(random.next())
        second = open_uniform(random.next())
        magnitude = math.sqrt(-2.0 * math.log(first))
        angle = 2.0 * math.pi * second
        values.append(magnitude * math.cos(angle) * standard_deviation)
        if len(values) < count:
            values.append(magnitude * math.sin(angle) * standard_deviation)
    return values


def self_check():
    # Published SplitMix64 seed-zero sequence checks the finalizer constants,
    # increment-before-mix order, and unsigned 64-bit masking independently.
    random = SplitMix64(0)
    expected = [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]
    assert [random.next() for unused in expected] == expected
    assert 0.0 < open_uniform(0) < 1.0
    assert 0.0 < open_uniform(MASK64) < 1.0
    assert SplitMix64(123).bounded(1) == 0
    first = initialized_prefix(7, 0, 25, 1)
    second = initialized_prefix(7, 0, 25, 2)
    assert first[0] == second[0]


def csv(values):
    return ",".join(str(value) for value in values)


def render_vectors():
    seed = 1337
    raw_random = SplitMix64(seed)
    raw = [raw_random.next() for unused in range(5)]
    training, validation = split_indices(60000, 5000, seed)
    epoch_one = epoch_indices(training, seed, 1)
    translation_indices = [0, 1, 12345, 59999]
    weight_specs = [
        ("conv1.weight", 0, 25),
        ("conv2.weight", 2, 150),
        ("fc1.weight", 4, 256),
        ("fc2.weight", 6, 120),
        ("fc3.weight", 8, 84),
    ]

    rejection_bound = (1 << 63) + 1
    rejection_seed = 0
    while True:
        candidate = SplitMix64(rejection_seed)
        first_raw = candidate.next()
        second_raw = candidate.next()
        threshold = ((-rejection_bound) & MASK64) % rejection_bound
        if first_raw < threshold and second_raw >= threshold:
            break
        rejection_seed += 1

    lines = [
        "protocol=splitmix64-v1",
        "seed={0}".format(seed),
        "raw={0}".format(csv(raw)),
        "split_training_prefix={0}".format(csv(training[:10])),
        "split_validation_prefix={0}".format(csv(validation[:10])),
        "epoch1_shuffle_prefix={0}".format(csv(epoch_one[:10])),
        "rejection={0},{1},{2},{3},{4}".format(
            rejection_seed, rejection_bound, first_raw, second_raw,
            second_raw % rejection_bound),
    ]
    for index in translation_indices:
        dx, dy = translation(seed, 1, index)
        lines.append("translation.{0}={1},{2}".format(index, dx, dy))
    for name, ordinal, fan_in in weight_specs:
        values = initialized_prefix(seed, ordinal, fan_in, 5)
        lines.append("initial.{0}={1}".format(
            name, ",".join(format(value, ".17g") for value in values)))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    self_check()
    contents = render_vectors()
    assert contents == render_vectors()
    with open(arguments.output, "w") as output:
        output.write(contents)


if __name__ == "__main__":
    main()
