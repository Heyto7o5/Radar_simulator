"""Model-aware preflight used by the installed CLI and unified API."""


def inspect_configuration(raw):
    from .unified.applicability import inspect_configuration as inspect_dynamic
    from .unified.session import session_from_config
    if not isinstance(raw, dict):
        return {'status': 'unsupported', 'checks': [{'code': 'configuration', 'status': 'unsupported',
                                                   'message': 'configuration must be an object'}]}
    static_or_statistical = (raw.get('waveform') == 'sfcw' or raw.get('kind') in (
        'statistical_background', 'block_texture_background', 'legacy_separable_if'))
    if not static_or_statistical:
        return inspect_dynamic(raw)
    try:
        session = session_from_config(raw)
    except (ValueError, TypeError) as exc:
        return {'status': 'unsupported', 'checks': [{'code': 'configuration', 'status': 'unsupported',
                                                   'message': str(exc)}]}
    return {'status': 'usable', 'checks': [{'code': 'model_schema', 'status': 'usable',
        'message': 'Selected acquisition schema and resource checks passed; no synthesis performed.'}],
        'resources': session.estimate, 'data_origin': session.data_origin,
        'limitations': ['Usable is not measured fidelity.',
                       'Static spectrum, range-gated IQ and dynamic ADC are different physical domains.']}
