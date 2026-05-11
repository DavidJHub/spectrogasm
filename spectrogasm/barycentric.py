"""Barycentric radial-velocity correction (self-contained, no astropy).

The pipeline brings the stellar spectrum to a frame where the *telluric*
O2 lines are at rest, which is the **topocentric** frame (observer at
rest on Earth's surface). The radial velocity measured from
``lambda_rest`` is therefore topocentric. To compare against catalogue
values, which are reported in the **barycentric / heliocentric** frame,
one has to add the projection of the observer's velocity in the
barycentric frame onto the line of sight to the target:

    v_helio = v_topo + v_bary_correction
    v_bary_correction = -(v_observer_in_bary_frame . n_hat_target)

This module implements that calculation from scratch with the
low-precision algorithms of Meeus, *Astronomical Algorithms* (2nd ed.,
1998, chapters 7, 12, 25). Expected accuracy is ~0.5 km/s, well below
the per-line scatter in our data; if higher precision is ever needed,
swap in ``astropy.coordinates.SkyCoord.radial_velocity_correction``.

Components included:

* Earth's heliocentric orbital velocity (circular-orbit approximation
  on top of Meeus's apparent solar longitude).
* Observer's diurnal rotation about the Earth's axis.

Components omitted (each below ~50 m/s for our use):

* Earth's orbital eccentricity correction to the speed (max ~0.5 km/s).
* Aberration / nutation of the obliquity.
* Relativistic Doppler corrections.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta


# Mean Earth orbital speed (km/s). 2*pi*AU / sidereal year.
EARTH_ORBITAL_SPEED_KMS = 29.78474
# Equatorial surface rotation speed (km/s).
EARTH_ROT_SPEED_EQ_KMS = 0.46512


# --- Time helpers ---------------------------------------------------------

def julian_date(dt: datetime) -> float:
    """Julian Date for a UTC ``datetime``. Meeus, ch. 7."""
    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    day_frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    return (
        int(365.25 * (y + 4716))
        + int(30.6001 * (m + 1))
        + dt.day + day_frac + b - 1524.5
    )


def _T_centuries(jd: float) -> float:
    """Julian centuries since J2000.0."""
    return (jd - 2451545.0) / 36525.0


# --- Sun / Earth ----------------------------------------------------------

def _sun_apparent_longitude_rad(jd: float) -> float:
    """Sun's apparent ecliptic longitude (rad). Meeus, ch. 25, low precision."""
    T = _T_centuries(jd)
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360.0
    M_deg = (357.52911 + 35999.05029 * T - 0.0001537 * T * T) % 360.0
    M = math.radians(M_deg)
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(M)
        + (0.019993 - 0.000101 * T) * math.sin(2 * M)
        + 0.000289 * math.sin(3 * M)
    )
    return math.radians((L0 + C) % 360.0)


def _earth_velocity_ecliptic_kms(jd: float) -> tuple[float, float, float]:
    """Earth's velocity in the barycentric ecliptic frame (km/s).

    Circular-orbit approximation: Earth is at heliocentric ecliptic
    longitude ``lambda_sun + 180`` deg, moving tangent to the orbit in
    the prograde direction (+90 deg from its position vector).
    """
    lam_sun = _sun_apparent_longitude_rad(jd)
    # Earth heliocentric longitude is lam_sun + pi; velocity is +pi/2 from
    # that, hence lam_sun + 3*pi/2.
    angle = lam_sun + 1.5 * math.pi
    v_x = EARTH_ORBITAL_SPEED_KMS * math.cos(angle)
    v_y = EARTH_ORBITAL_SPEED_KMS * math.sin(angle)
    return v_x, v_y, 0.0


def _obliquity_rad(jd: float) -> float:
    """Mean obliquity of the ecliptic (rad). Meeus eq. 22.2 (truncated)."""
    T = _T_centuries(jd)
    return math.radians(23.439291 - 0.0130042 * T - 1.64e-7 * T * T)


def _ecliptic_to_equatorial(vec: tuple[float, float, float], jd: float):
    """Rotate an ecliptic-frame vector to equatorial coordinates."""
    x, y, z = vec
    eps = _obliquity_rad(jd)
    ce, se = math.cos(eps), math.sin(eps)
    return x, ce * y - se * z, se * y + ce * z


# --- Diurnal rotation -----------------------------------------------------

def _gmst_hours(jd: float) -> float:
    """Greenwich Mean Sidereal Time (hours). Meeus eq. 12.4."""
    T = _T_centuries(jd)
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - T * T * T / 38710000.0
    ) % 360.0
    return gmst_deg / 15.0


def _hour_angle_rad(jd: float, lon_deg: float, ra_deg: float) -> float:
    """Local hour angle of the target (rad, positive west of meridian)."""
    lst_hours = (_gmst_hours(jd) + lon_deg / 15.0) % 24.0
    H_hours = lst_hours - ra_deg / 15.0
    # Wrap to (-12, +12] hours so the small-angle interpretation holds.
    H_hours = ((H_hours + 12.0) % 24.0) - 12.0
    return math.radians(H_hours * 15.0)


# --- Public API -----------------------------------------------------------

def target_unit_vector_equatorial(ra_deg: float, dec_deg: float):
    """Unit vector from observer to target in the equatorial frame."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return (
        math.cos(dec) * math.cos(ra),
        math.cos(dec) * math.sin(ra),
        math.sin(dec),
    )


def barycentric_correction_kms(
    obs_time_utc: datetime,
    target_ra_deg: float,
    target_dec_deg: float,
    obs_lat_deg: float,
    obs_lon_deg: float,
) -> float:
    """Return the km/s to ADD to a topocentric v_rad to get v_heliocentric.

    Convention matches ``astropy.SkyCoord.radial_velocity_correction``::

        v_helio = v_topo + barycentric_correction_kms(...)
    """
    jd = julian_date(obs_time_utc)
    n_hat = target_unit_vector_equatorial(target_ra_deg, target_dec_deg)

    # Orbital part: Earth's velocity in barycentric, projected on n_hat.
    v_orb_ecl = _earth_velocity_ecliptic_kms(jd)
    v_orb_eq = _ecliptic_to_equatorial(v_orb_ecl, jd)
    v_obs_dot_n = sum(a * b for a, b in zip(v_orb_eq, n_hat))

    # Diurnal part: observer's eastward rotation projected on n_hat.
    # East unit vector in equatorial coords at the observer is
    # (-sin(LST), +cos(LST), 0), and its dot product with n_hat
    # simplifies to -cos(dec) * sin(H), where H is the hour angle.
    H = _hour_angle_rad(jd, obs_lon_deg, target_ra_deg)
    v_rot_speed = EARTH_ROT_SPEED_EQ_KMS * math.cos(math.radians(obs_lat_deg))
    v_rot_dot_n = -v_rot_speed * math.cos(math.radians(target_dec_deg)) * math.sin(H)

    return -(v_obs_dot_n + v_rot_dot_n)


def bogota_observation_time_utc(date_iso: str, local_hour: int = 19) -> datetime:
    """UTC ``datetime`` for the night observed at Bogotá ~local_hour PM.

    Bogotá is UTC-5 year-round (no DST). 19:00 local on a given date
    is therefore 00:00 UTC on the *following* calendar day. The dates
    in ``manifest.NIGHTS`` use the local (Bogotá) night-start date, so
    we shift forward by 1 day.
    """
    base = datetime.fromisoformat(date_iso)
    utc_hour = (local_hour + 5) % 24
    day_offset = (local_hour + 5) // 24
    return datetime(base.year, base.month, base.day, utc_hour) + timedelta(days=day_offset)
