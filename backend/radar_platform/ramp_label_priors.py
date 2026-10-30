"""Declared relative amplitude priors for label-only scenes, NOT measured RCS."""


def relative_class_amplitude(class_id: int) -> float:
    return {0: .25, 80: .45, 2: 1., 3: .65, 5: 2., 7: 2.4}.get(class_id, .8)
