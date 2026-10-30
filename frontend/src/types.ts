export interface RadarConfig {
  carrier_hz: number;
  bandwidth_hz: number;
  sample_rate_hz: number;
  pulse_width_s: number;
  prf_hz: number;
  pulses_per_cpi: 32 | 64 | 128;
  beam_half_angle_deg: number;
}

export interface PlatformConfig {
  position_ned_m: [number, number, number];
  speed_mps: number;
  dive_angle_deg: number;
  heading_deg: number;
}

export interface TargetConfig {
  id: string;
  position_ned_m: [number, number, number];
  amplitude: number;
}

export interface SimulationRequest {
  radar: RadarConfig;
  platform: PlatformConfig;
  scene: {
    targets: TargetConfig[];
    simplified_clutter_enabled: boolean;
    seed: number;
  };
  sequence: {
    frame_count: number;
    frame_interval_s: number;
  };
}

export interface DerivedMetrics {
  wavelength_m: number;
  range_resolution_m: number;
  range_sample_spacing_m: number;
  max_unambiguous_range_m: number;
  full_echo_max_range_m: number;
  max_unambiguous_velocity_mps: number;
  velocity_bin_mps: number;
  pri_s: number;
  cpi_s: number;
  num_fast_time_samples: number;
  warnings: string[];
}

export interface TargetValidation {
  target_id: string;
  visible: boolean;
  in_beam: boolean;
  true_range_m: number;
  raw_detected_range_m: number | null;
  range_doppler_bias_m: number;
  detected_range_m: number | null;
  range_error_m: number | null;
  true_radial_velocity_mps: number;
  true_doppler_hz: number;
  aliased_doppler_hz: number;
  ambiguity_order: number;
  apparent_velocity_mps: number;
  detected_apparent_velocity_mps: number | null;
  doppler_bin_error: number | null;
  passed: boolean;
}

export interface FrameMetadata {
  frame_index: number;
  frame_time_s: number;
  elapsed_s: number;
  platform_position_ned_m: [number, number, number];
  display_rows: number;
  display_cols: number;
  range_start_m: number;
  range_step_m: number;
  doppler_start_hz: number;
  doppler_step_hz: number;
  velocity_start_mps: number;
  velocity_step_mps: number;
  range_profile_db: number[];
  doppler_profile_db: number[];
  validations: TargetValidation[];
}

export interface JobState {
  job_id: string;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  progress: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  derived_metrics: DerivedMetrics;
  warnings: string[];
  frames: FrameMetadata[];
}

export interface BinaryMatrix {
  data: Float32Array;
  rows: number;
  cols: number;
}

export interface RampStatus {
  configured: boolean;
  root: string | null;
  is_real_available: boolean;
  real_frame_count_preview: number;
  count_is_capped?: boolean;
  source_url: string;
  radar_config: Record<string, number>;
}

export interface RampLabel {
  uid: string;
  class_id: number;
  x_m: number;
  y_m: number;
  width_m: number;
  length_m: number;
  range_m: number;
  radial_velocity_mps: number | null;
  nearest_strong_peak_range_m: number | null;
  range_difference_m: number | null;
}

export interface RampResult {
  result_id: string;
  is_real: boolean;
  source_label: string;
  frame_id: string | null;
  adc_shape: number[];
  rd_shape: number[];
  range_start_m: number;
  range_step_m: number;
  velocity_start_mps: number;
  velocity_step_mps: number;
  labels: RampLabel[];
  channel_names: string[];
  radar_config: Record<string, number>;
  source_url: string;
  validation_scope: string[];
}

export interface SimulationPreset {
  name: string;
  description: string;
  request: SimulationRequest;
  derived_metrics: DerivedMetrics;
  warnings: string[];
}

export interface TerrainGrid {
  id: string;
  name: string;
  coordinate_frame: string;
  resolution_m: number;
  x_m: number[];
  y_m: number[];
  elevation_m: number[][];
  is_physical_scattering_calibrated: boolean;
}
