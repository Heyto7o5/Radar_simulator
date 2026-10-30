"""Angle-dependent empirical land strength, with no out-of-coverage clamping."""
import numpy as np

def land_amplitude_factor(scene,subset,ct,cr,dt,dr,rt,rr,illuminated,carrier_hz,directional):
    """Replace geometric cosine taper by sqrt(sigma0) on empirical ground cells.

    Existing diffuse cell amplitudes already contain sqrt(surface area). Adding
    another Lambert cosine here would count angular response a second time.
    No physical terrain texture, vegetation, receiver gain or full-wave claim.
    """
    if not scene.ground_models:return directional
    result=np.array(directional,copy=True)
    object_indices=scene.object_index[subset]
    for object_id in np.unique(object_indices):
        if int(object_id) not in scene.ground_models:continue
        curve,spec=scene.ground_models[int(object_id)]
        nodes=object_indices==object_id
        visible=nodes[:,None]&illuminated&(ct>0)&(cr>0)
        result[nodes]=0.
        if not visible.any():continue
        bistatic=np.rad2deg(np.arccos(np.clip(np.sum(dt*dr,axis=-1)/(rt*rr),-1,1)))
        if np.any(bistatic[visible]>spec.max_bistatic_angle_deg):
            raise ValueError('empirical monostatic land coefficients cannot cover this bistatic geometry')
        incidence=.5*np.rad2deg(np.arccos(np.clip(ct,-1,1))+np.arccos(np.clip(cr,-1,1)))
        labels=scene.land_cover_labels[subset] if scene.land_cover_labels is not None else np.full(len(object_indices),spec.land_cover)
        for label in np.unique(labels[nodes]):
            selected=visible&(labels==label)[:,None]
            if not selected.any():continue
            try:
                sigma=curve.predict_db(carrier_hz,incidence[selected],spec.polarization,str(label))
            except ValueError as exc:
                raise ValueError(f'illuminated ground outside measured coefficient coverage or declared table coverage: {exc}') from exc
            result[selected]=10**(sigma/20)
    return result
