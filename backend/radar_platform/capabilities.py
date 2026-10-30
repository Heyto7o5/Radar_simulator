"""Machine-readable feature index; availability is not measured fidelity."""


def capabilities():
    return {
        'workflow': ['configuration', 'preflight', 'session', 'physical_model', 'complex_products', 'comparison'],
        'entry_points': {'cli': 'radar-platform', 'http': '/api/v1/scene-simulations',
                         'ui': '/simulate', 'python': 'radar_platform.runner.run_configuration'},
        'models': [
            {'id': 'pulsed_lfm', 'domain': 'adc', 'products': ['adc', 'rd_channels', 'rad', 'rd_power', 'ra_power']},
            {'id': 'fmcw', 'domain': 'adc', 'products': ['adc', 'rd_channels', 'rad', 'rd_power', 'ra_power']},
            {'id': 'sfcw', 'domain': 'static_frequency_response', 'products': ['spectrum']},
            {'id': 'statistical_background', 'domain': 'range_gated_iq', 'products': ['iq', 'rd']},
            {'id': 'block_texture_background', 'domain': 'historical_range_gated_iq', 'products': ['iq', 'rd']},
            {'id': 'legacy_separable_if', 'domain': 'historical_separable_if', 'products': ['if', 'rd']},
        ],
        'dynamic_features': ['points', 'clouds', 'plates', 'boxes', 'triangle_meshes', 'DEM',
            'relative_motion', 'sampled_platform_trajectory', 'SISO', 'SIMO', 'TDM_MIMO',
            'ULA_URA_custom_geometry', 'RA_RAD', 'transmit_subaperture', 'element_patterns',
            'electronic_scan', 'surface_occlusion', 'land_scattering_tables',
            'ADC_white_colored_noise', 'range_background', 'applicability', 'motion_error_checks'],
        'data_adapters': ['RAMP', 'MAROON', 'IPIX', 'Maqu', 'TSMS', 'DART', 'Copernicus_DEM', 'WorldCover'],
        'optional_applications': ['fixed_scene_RDA_inference', 'navigation_experiments', 'multi_DEM_learning'],
        'boundaries': ['Features are not universally composable; schema and preflight govern supported combinations.',
            'Static spectra are not RD; range-gated IQ is not ADC.',
            'Analytic tests and real geometry do not establish measured high-altitude fidelity.',
            'No general full-wave scattering, calibrated absolute power or universal environment multipath.',
            'Single-PRF ambiguity is reported, not generally resolved.',
            'Application experiments consume the shared simulator, not alternate echo engines.',
            'Historical approximation models are retained for reproducibility, not new physical scenes.'],
    }
