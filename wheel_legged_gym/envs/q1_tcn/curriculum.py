"""Nominal evaluation gate for expanding command support without changing reward."""
import math


def tracking_gate(tracking, standing, thresholds):
    """Return explicit pass/fail checks; missing/invalid observations fail."""
    def value(result,key):
        metric=result.get('metrics',{}).get(key,{})
        number=metric.get('mean')
        if not metric.get('n') or number is None or not math.isfinite(number):
            return float('inf')
        return number

    checks={
        'tracking_survival':value(tracking,'failed')==0,
        'standing_survival':value(standing,'failed')==0,
        'steady_speed':value(tracking,'steady_speed_rmse_m_s')<=thresholds['steady_rmse'],
        'forward_bias':abs(value(tracking,'forward_speed_bias_m_s'))<=thresholds['bias'],
        'reverse_bias':abs(value(tracking,'reverse_speed_bias_m_s'))<=thresholds['bias'],
        'settling':value(tracking,'command_settled_fraction')>=thresholds['settled_fraction'],
        'stop_speed':value(tracking,'stop_speed_rmse_m_s')<=thresholds['stop_rmse'],
        'stand_position':value(standing,'locked_parking_end_m')<=thresholds['stand_drift'],
        'stand_pitch':value(standing,'pitch_rms_rad')<=math.radians(thresholds['stand_tilt_deg']),
        'stand_roll':value(standing,'roll_rms_rad')<=math.radians(thresholds['stand_tilt_deg']),
    }
    # Missing settling data must not pass because +inf exceeds the minimum.
    settled=value(tracking,'command_settled_fraction')
    checks['settling']=math.isfinite(settled) and settled>=thresholds['settled_fraction']
    n=tracking.get('trials',0)
    for key in ['steady_speed_rmse_m_s','forward_speed_bias_m_s','reverse_speed_bias_m_s','stop_speed_rmse_m_s']:
        checks['complete_'+key]=n>0 and tracking.get('metrics',{}).get(key,{}).get('n')==n
    return all(checks.values()),checks
