export type Vec3 = [number, number, number];
export interface Waveform {
  kind: "pulsed_lfm" | "fmcw";
  carrier_hz?: number; bandwidth_hz?: number; pulse_width_s?: number;
  start_hz?: number; slope_hz_s?: number; ramp_s?: number;
  sample_rate_hz: number; slot_period_s: number; adc_start_s: number; samples: number; loops: number;
}
export interface SceneObject {
  scattering_model?: "centroid" | "scalar_po"; land_scattering?: Record<string, unknown> | null;
  id: string; kind: "point" | "box" | "plate" | "cloud" | "mesh";
  position_ned_m: Vec3; velocity_ned_mps: Vec3;
  amplitude?: number; phase_rad?: number; relative_reflectivity?: number;
  size_m?: number[]; divisions?: number; normal_ned?: Vec3; yaw_deg?: number;
  scattering?: "diffuse" | "coherent"; two_sided?: boolean;
  points_local_m?: Vec3[]; amplitudes?: number[]; phases_rad?: number[];
  vertices_local_m?: Vec3[]; faces?: Vec3[];
}
export interface Terrain {
  max_grid_vertices?: number; land_cover_map?: string[][] | null; land_scattering?: Record<string, unknown> | null;
  id: string; kind: "flat" | "hills" | "dem";
  north_bounds_m: [number, number]; east_bounds_m: [number, number]; spacing_m: number;
  base_elevation_m: number; relative_reflectivity: number; scattering: "diffuse" | "coherent";
  hills: { center_ne_m: [number, number]; height_m: number; sigma_ne_m: [number, number] }[];
  north_m?: number[] | null; east_m?: number[] | null; elevation_up_m?: number[][] | null;
}
export interface SceneRequest {
  input_provenance?: string[];
  visibility?: { occlusion_enabled: boolean; report_coverage: boolean; occluders: "terrain" | "all_surfaces"; accelerator?: "brute_force" | "bvh"; batch_bvh?: boolean; endpoint_epsilon_m?: number; max_ray_triangle_tests?: number };
  radiation?: Radiation | null;
  receiver_noise?: { variance: number; seed: number; variance_per_rx_tx?: number[][] | null; fast_time_spectral_power?: number[] | null; provenance?: string | null } | null;
  range_background?: { power_scale: number; provenance: string; seed: number; channel_mode?: "independent" | "steered"; model: Record<string, unknown>; [key: string]: unknown } | null;
  save_components?: boolean;
  schema_version: "1.0"; name: string; waveform: Waveform;
  platform: { trajectory?: { time_s: number[]; position_ned_m: Vec3[]; velocity_ned_mps: Vec3[]; quaternion_xyzw: number[][]; provenance: string } | null; beam_model?: string; beam_taper_deg?: number; position_ned_m: Vec3; velocity_ned_mps: Vec3; boresight_ned: Vec3; right_hint_ned: Vec3; beam_half_angle_deg: number };
  array: { layout: "siso" | "ula" | "ura" | "custom"; num_tx: number; num_rx: number; spacing_m: number | null;
    tx_local_m?: Vec3[] | null; rx_local_m?: Vec3[] | null; tx_order?: number[] | null; tx_offsets_s?: number[] | null; loop_period_s?: number | null };
  scene: { allow_empty?: boolean; objects: SceneObject[]; terrain: Terrain | null; seed: number; amplitude_reference_range_m: number; max_scatterers: number };
  processing: { range_limits_m: [number, number]; range_window: string; doppler_window: string; spatial_window: string;
    azimuth_deg: number[]; elevation_deg: number[]; tdm_compensation: boolean; range_fft_size?: number | null;
    doppler_fft_size?: number | null; save_adc: boolean; max_peaks: number };
  sequence: { frame_count: number; frame_interval_s: number; start_time_s: number };
}
export interface Features {
  terrain: boolean; array: boolean; platformMotion: boolean; targetMotion: boolean; disabledObjects: string[];
}
export interface GeometryObject { id: string; kind: string; initial_center_ned_m: Vec3; velocity_ned_mps: Vec3; bounds_ned_m: [Vec3, Vec3]; scatterers: number }
export interface SceneGeometry {
  velocities_ned_mps?: Vec3[]; land_cover_labels?: string[]; scatterer_count?: number;
  positions_ned_m: Vec3[]; object_index: number[]; objects: GeometryObject[];
  terrain: { north_m: number[]; east_m: number[]; elevation_up_m: number[][]; kind: string; id: string } | null;
  preview_stride: number;
}
export interface Preview {
  state?: Auxiliary;
  request: SceneRequest; notes: string[]; geometry: SceneGeometry;
  estimate: { estimated_output_bytes: number; scatterers: number };
  array: { virtual_local_m: Vec3[]; tx_local_m: Vec3[]; rx_local_m: Vec3[]; loop_period_s: number };
  derived: { carrier_hz: number; range_resolution_m: number; max_unambiguous_velocity_mps: number; velocity_cell_mps: number; samples: number; loops: number; channels: number };
}
export interface SceneFrame {
  truth_epoch_s?: number; processing?: Record<string, unknown>; visibility?: { coverage?: Coverage };
  frame_index: number; frame_start_s: number; elapsed_s: number; platform_position_ned_m: Vec3; warnings: string[];
  axes: { range_m: number[]; velocity_mps: number[]; closing_doppler_hz: number[]; azimuth_deg: number[]; elevation_deg: number[] };
  array: { channel_map: { channel: number; rx: number; tx: number }[]; two_dimensional_angle_geometry: boolean };
  peak_candidates: { range_bin: number; doppler_bin: number; range_m: number; apparent_closing_velocity_mps: number }[];
  object_truth: { id: string; center_ned_m: Vec3; range_extent_m?: number[]; closing_velocity_extent_mps?: number[]; [key: string]: unknown }[];
  products: Record<string, { shape: number[]; dtype: string; file: string }>;
}
export interface SceneJob { created_at?: string; estimate?: { applicability?: CheckReport; [key: string]: unknown }; job_id: string; name: string; status: "queued" | "running" | "complete" | "failed" | "cancelled"; error: string | null; progress: number; frames: SceneFrame[]; frame_count: number }
export interface Matrix { data: Float32Array; rows: number; cols: number; reference: number }
export interface CatalogFrame { id: string; name: string; frame_index: number; camera_available: boolean }
export interface Catalog { configured: boolean; available: boolean; frames?: number; source_url: string; sequences: { id: string; name: string; count: number; frames: CatalogFrame[] }[] }
export interface RealLabel { uid: string; class_id: number; class_name: string; x_m: number; y_m: number; width_m: number; length_m: number; range_m: number; radial_velocity_mps: number | null; tracking_ambiguous: boolean }
export interface ComparisonRecord {
  record_id: string; frame_id: string; source_frame: string; sequence: string; frame_index: number; is_real: true;
  camera_available: boolean; camera_url: string | null; labels: RealLabel[]; template: SceneRequest | null;
  reference_power: number; range_m: number[]; velocity_mps: number[]; channels: string[];
  amplitude_note: string; target_note: string; acquisition: Pick<SceneRequest, "waveform" | "platform" | "array" | "processing">;
}

export function clone<T>(value: T): T { return JSON.parse(JSON.stringify(value)) as T; }
export function defaultFeatures(request: SceneRequest): Features {
  return { terrain: Boolean(request.scene.terrain), array: request.array.layout === "custom" || request.array.num_tx * request.array.num_rx > 1,
    platformMotion: true, targetMotion: true, disabledObjects: [] };
}
export function effectiveRequest(request: SceneRequest, features: Features): SceneRequest {
  const value = clone(request);
  if (!features.terrain) value.scene.terrain = null;
  value.scene.objects = value.scene.objects.filter(o => !features.disabledObjects.includes(o.id));
  if (!features.platformMotion) value.platform.velocity_ned_mps = [0, 0, 0];
  if (!features.targetMotion) value.scene.objects.forEach(o => { o.velocity_ned_mps = [0, 0, 0]; });
  if (!features.array && request.array.num_tx * request.array.num_rx > 1) {
    value.array = { layout: "siso", num_tx: 1, num_rx: 1, spacing_m: null };
    value.processing.azimuth_deg = [0]; value.processing.elevation_deg = [0];
  }
  return value;
}
export const objectNames: Record<SceneObject["kind"], string> = { point: "点目标", box: "长方体", plate: "平板", cloud: "点云", mesh: "网格" };
export function objectLabel(id: string) {
  const builtins: Record<string, string> = { building: "简化建筑", moving_target: "运动点目标", reference_plate: "参考板", terrain: "地形" };
  if (builtins[id]) return builtins[id];
  const ambiguous = id.match(/^(.+)@untracked:\d+:(\d+)$/);
  if (ambiguous) return `目标 ${ambiguous[1]}-${Number(ambiguous[2]) + 1}`;
  return /^\d+$/.test(id) ? `目标 ${id}` : id;
}

export interface Pattern { kind: "isotropic" | "cosine_power" | "axisymmetric_table"; power_exponent?: number; angle_deg?: number[] | null; power_db?: number[] | null; frequency_min_hz?: number | null; frequency_max_hz?: number | null; evidence?: string; source?: string }
export interface Radiation { tx_element: Pattern; rx_element: Pattern; replace_legacy_cone: boolean; transmit_aperture?: { positions_local_m: Vec3[]; amplitudes?: number[] | null; phases_rad?: number[] | null; steering_azimuth_deg: number; steering_elevation_deg: number; scan_schedule: { time_s: number; azimuth_deg: number; elevation_deg: number }[]; max_wideband_phase_error_rad?: number } | null }
export interface CheckReport { status: "usable" | "risk" | "unsupported"; checks: { code: string; status: string; message: string; metrics?: Record<string, unknown> }[]; limitations?: string[]; motion_check?: { status: string; [key: string]: unknown }; motion_error?: string; resources?: Record<string, number> }
export interface Coverage { positions_at_frame_start_ned_m: Vec3[]; object_index: number[]; objects: string[]; illuminated_tx_slots: number[]; unblocked_channel_slots: number[]; sampled_echo_channel_slots: number[]; tx_slot_denominator: number; channel_slot_denominator: number; occlusion_evaluated: boolean; positions_epoch_s: number; preview_stride: number; total_scatterers: number }
export interface Auxiliary { time_s: number; platform_position_ned_m: Vec3; platform_velocity_ned_mps: Vec3; basis_ned: number[][]; coverage: Coverage | null; pattern: { directions_ned: Vec3[]; normalized_power: number[]; rows: number; cols: number; time_s: number } | null; array: Preview["array"]; checks?: CheckReport; objects: { id: string; position_ned_m: Vec3 }[] }
export interface ExperimentSummary { id: string; name: string; status: SceneJob["status"]; frames: number; frame_count: number; created_at: string; saved: boolean; archived: boolean; error?: string }
export interface Experiment { job: SceneJob; request: SceneRequest; config_hash: string; saved?: { title: string; display: Record<string, unknown> } | null }
export interface CaseSpec { id: string; title: string; category: string; description: string; steps: string[]; boundary: string; light: boolean; archive_id?: string; available?: boolean }
export interface DisplayData { power: number[][]; axes: SceneFrame["axes"]; frame_start_s: number; channel: string; units: string; config_hash: string; processing: Record<string, unknown> }
