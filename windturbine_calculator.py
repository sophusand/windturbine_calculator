import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import json
from datetime import datetime

st.set_page_config(page_title="Vindmølle Beregner", layout="wide")

st.title("Interaktiv Vindmølle Beregner")
st.markdown("Beregn mekanisk og elektrisk effekt for forskellige vindmølletyper")

# Konstanter
RHO_DEFAULT = 1.225  # lufttæthed kg/m³ (normalt lufttryk ved havniveau)
PI = np.pi

# KOMMERCIELLE MØLLER DATABASE
commercial_turbines = {
    "Vestas V90": {"radius": 45, "cp": 0.45, "power_rated": 3000},
    "GE 2.85-127": {"radius": 63.5, "cp": 0.42, "power_rated": 2850},
    "Siemens SWT-6.0-154": {"radius": 77, "cp": 0.44, "power_rated": 6000},
    "Enercon E126": {"radius": 63, "cp": 0.48, "power_rated": 7500},
    "DIY Hobby": {"radius": 0.3, "cp": 0.20, "power_rated": 50},
    "DIY Moderne": {"radius": 1.0, "cp": 0.45, "power_rated": 500},
}

DEFAULTS = {
    "calculation_history": [],
    "designer_radius": 0.3,
    "designer_cp_hobby": 0.20,
    "designer_result_radius": None,
    "designer_result_cp": None,
    "designer_cp_hobby_pct": 20,
    "motor_efficiency_pct": 80,
    "tip_speed_ratio": 3.2,
    "kv_value": 0.0074,
    "gear_ratio": 2.0,
    "v_drop": 0.64,
    "rho": RHO_DEFAULT,
    "blade_mass": 1.0,
    "blade_cm_radius": 0.15,
    "blade_chord": 0.05,
    "num_blades": 3,
    "installation_height": 10.0,
    "yaw_angle": 0,
    "pitch_angle": -2,
    "generator_power_rated": 500,
    "weibull_k": 2.0,
    "v_min": 1.0,
    "v_max": 15.0,
    "use_log_scale": False,
    "compare_turbines": False,
    "turbine_type": "Moderne vindmølle",
    "selected_commercial": list(commercial_turbines.keys())[0],
    "selected_wind": 1.0,
}

def init_session_state(defaults):
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, (list, dict)):
                st.session_state[key] = value.copy()
            else:
                st.session_state[key] = value

def apply_settings(settings, defaults):
    allowed_keys = set(defaults.keys()) - {
        "calculation_history",
        "designer_result_radius",
        "designer_result_cp",
    }
    for key in allowed_keys:
        if key in settings:
            value = settings[key]
            if key == "turbine_type" and value not in {"Moderne vindmølle", "Lille hobbymølle"}:
                continue
            if key == "selected_commercial" and value not in commercial_turbines:
                continue
            if key == "designer_cp_hobby_pct":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                value = max(10, min(100, value))
                st.session_state[key] = value
                st.session_state["designer_cp_hobby"] = value / 100
                continue
            st.session_state[key] = value

init_session_state(DEFAULTS)

# Sidebar - Input parametre
st.sidebar.header("Input Parametre")

# Session state for designer updates er initialiseret via DEFAULTS

# Blade radius - use temporary result if available
slider_radius_value = st.session_state.designer_result_radius if st.session_state.designer_result_radius else st.session_state.designer_radius

radius = st.sidebar.slider(
    "Vinge-radius (r) [m]",
    min_value=0.1,
    max_value=2.0,
    value=slider_radius_value,
    step=0.1,
    key="designer_radius"
)

# Validering af radius
if radius < 0.15:
    st.sidebar.warning("Advarsel: Meget lille radius kan give urealistiske resultater")

st.sidebar.subheader("Effektkoefficienter (Cp)")
# Betz grænse er konstant på 59.3%
cp_betz = 0.593

# Moderne vindmølle er konstant på 45%
cp_modern = 0.45

# Use temporary result if available
slider_cp_value = st.session_state.designer_result_cp if st.session_state.designer_result_cp else st.session_state.designer_cp_hobby

cp_hobby = st.sidebar.slider(
    "Lille hobbymølle (%)",
    min_value=10,
    max_value=100,
    value=max(10, min(100, int(round(slider_cp_value * 100)))),
    step=1,
    key="designer_cp_hobby_pct"
) / 100

# Gemte cp_hobby værdi til designer state
st.session_state.designer_cp_hobby = cp_hobby

st.sidebar.subheader("Elektriske Konstanter")
motor_efficiency = st.sidebar.slider(
    "Motors MPP nyttevirkning (%)",
    min_value=10,
    max_value=99,
    value=int(st.session_state.motor_efficiency_pct),
    step=1,
    key="motor_efficiency_pct"
) / 100

if motor_efficiency < 0.5:
    st.sidebar.warning("Advarsel: Meget lav motoreffektivitet (under 50%)")

tip_speed_ratio = st.sidebar.slider(
    "Tip Speed Ratio (λ)",
    min_value=1.0,
    max_value=10.0,
    value=float(st.session_state.tip_speed_ratio),
    step=0.1,
    key="tip_speed_ratio"
)

kv = st.sidebar.number_input(
    "Motorkonstant (kv) [V/RPM]",
    min_value=0.001,
    max_value=0.02,
    value=float(st.session_state.kv_value),
    step=0.0001,
    format="%.4f",
    key="kv_value"
)

gear_ratio = st.sidebar.slider(
    "Gearudveksling",
    min_value=1.0,
    max_value=10.0,
    value=float(st.session_state.gear_ratio),
    step=0.5,
    key="gear_ratio"
)

v_drop = st.sidebar.slider(
    "Diode spændingsfald (V)",
    min_value=0.1,
    max_value=1.5,
    value=float(st.session_state.v_drop),
    step=0.01,
    key="v_drop"
)

# Avancerede indstillinger
st.sidebar.subheader("Avancerede Indstillinger")
with st.sidebar.expander("Lufttæthed og andre parametre"):
    rho = st.number_input(
        "Lufttæthed (ρ) [kg/m³]",
        min_value=0.9,
        max_value=1.5,
        value=float(st.session_state.rho),
        step=0.01,
        format="%.3f",
        key="rho"
    )
    
    # Blade massa for centrifugal kraft
    blade_mass = st.number_input(
        "Blade masse [kg]",
        min_value=0.1,
        max_value=10.0,
        value=float(st.session_state.blade_mass),
        step=0.1,
        key="blade_mass"
    )
    
    # Center of mass radius
    blade_cm_radius = st.number_input(
        "Blade massecenters radius [m]",
        min_value=0.05,
        max_value=2.0,
        value=float(st.session_state.blade_cm_radius),
        step=0.05,
        key="blade_cm_radius"
    )
    
    # Blade chord for solidity ratio
    blade_chord = st.number_input(
        "Blade chord (bredde) [m]",
        min_value=0.01,
        max_value=0.5,
        value=float(st.session_state.blade_chord),
        step=0.01,
        key="blade_chord"
    )
    
    num_blades = st.slider(
        "Antal blades",
        min_value=1,
        max_value=4,
        value=int(st.session_state.num_blades),
        step=1,
        key="num_blades"
    )
    
    # Højde for wind shear
    installation_height = st.number_input(
        "Installationshøjde [m]",
        min_value=1.0,
        max_value=100.0,
        value=float(st.session_state.installation_height),
        step=1.0,
        key="installation_height"
    )
    
    # Yaw vinkel
    yaw_angle = st.slider(
        "Yaw vinkel fra vind [°]",
        min_value=0,
        max_value=90,
        value=int(st.session_state.yaw_angle),
        step=5,
        key="yaw_angle"
    )
    
    # Pitch vinkel
    pitch_angle = st.slider(
        "Pitch vinkel [°]",
        min_value=-45,
        max_value=90,
        value=int(st.session_state.pitch_angle),
        step=1,
        key="pitch_angle"
    )
    
    # Generator specs
    generator_power_rated = st.number_input(
        "Generator nominaleffekt [W]",
        min_value=10,
        max_value=10000,
        value=int(st.session_state.generator_power_rated),
        step=50,
        key="generator_power_rated"
    )
    
    # Weibull k-faktor
    weibull_k = st.slider(
        "Weibull k-parameter (vindfordeling)",
        min_value=1.5,
        max_value=3.0,
        value=float(st.session_state.weibull_k),
        step=0.1,
        key="weibull_k"
    )

# Økonomisk beregning

# Vindhastighed interval
st.sidebar.subheader("Vindhastighed Interval")
col1, col2 = st.sidebar.columns(2)
with col1:
    v_min = st.number_input(
        "Fra [m/s]",
        min_value=0.1,
        value=float(st.session_state.v_min),
        step=0.5,
        key="v_min"
    )
with col2:
    v_max = st.number_input(
        "Til [m/s]",
        min_value=1.0,
        value=float(st.session_state.v_max),
        step=1.0,
        key="v_max"
    )

if v_min >= v_max:
    st.sidebar.error("Starthastighed må være mindre end sluthastighed")
    v_min = 1.0
    v_max = 15.0
    st.session_state.v_min = v_min
    st.session_state.v_max = v_max

# Grafindstillinger
st.sidebar.subheader("Grafindstillinger")
use_log_scale = st.sidebar.checkbox(
    "Brug logaritmisk skala",
    value=bool(st.session_state.use_log_scale),
    key="use_log_scale"
)
compare_turbines = st.sidebar.checkbox(
    "Sammenlign to møller",
    value=bool(st.session_state.compare_turbines),
    key="compare_turbines"
)

# Del/Importér indstillinger
st.sidebar.subheader("Del/Importér indstillinger")

settings_payload = {
    "designer_radius": float(radius),
    "designer_cp_hobby": float(cp_hobby),
    "designer_cp_hobby_pct": int(round(cp_hobby * 100)),
    "motor_efficiency_pct": int(round(motor_efficiency * 100)),
    "tip_speed_ratio": float(tip_speed_ratio),
    "kv_value": float(kv),
    "gear_ratio": float(gear_ratio),
    "v_drop": float(v_drop),
    "rho": float(rho),
    "blade_mass": float(blade_mass),
    "blade_cm_radius": float(blade_cm_radius),
    "blade_chord": float(blade_chord),
    "num_blades": int(num_blades),
    "installation_height": float(installation_height),
    "yaw_angle": int(yaw_angle),
    "pitch_angle": int(pitch_angle),
    "generator_power_rated": int(generator_power_rated),
    "weibull_k": float(weibull_k),
    "v_min": float(v_min),
    "v_max": float(v_max),
    "use_log_scale": bool(use_log_scale),
    "compare_turbines": bool(compare_turbines),
    "turbine_type": st.session_state.turbine_type,
    "selected_commercial": st.session_state.selected_commercial,
    "selected_wind": float(st.session_state.selected_wind),
}

config_json = json.dumps(settings_payload, ensure_ascii=False, indent=2)
st.sidebar.download_button(
    "Download indstillinger (JSON)",
    data=config_json,
    file_name="windturbine_settings.json",
    mime="application/json"
)

uploaded_settings = st.sidebar.file_uploader(
    "Importér indstillinger (JSON)",
    type=["json"],
    accept_multiple_files=False
)

if uploaded_settings is not None:
    try:
        imported_settings = json.load(uploaded_settings)
        apply_settings(imported_settings, DEFAULTS)
        st.sidebar.success("Indstillinger importeret. Genindlæser…")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Kunne ikke importere filen: {exc}")

if st.sidebar.button("Nulstil standarder"):
    for key, value in DEFAULTS.items():
        if isinstance(value, (list, dict)):
            st.session_state[key] = value.copy()
        else:
            st.session_state[key] = value
    st.sidebar.success("Nulstillet. Genindlæser…")
    st.rerun()

# AVANCEREDE FUNKTIONER
def validate_inputs(radius, cp_val, tsr, motor_eff):
    """Validér input parametre og giv advarsler"""
    warnings = []
    
    if cp_val > 0.593:
        warnings.append(f"⚠️ Cp={cp_val:.3f} overstiger Betz græns (59.3%)")
    
    if radius < 0.2 and tsr > 5:
        warnings.append(f"⚠️ Lille radius med høj TSR kan give urealistiske RPM")
    
    if motor_eff < 0.6:
        warnings.append(f"⚠️ Motoreffektivitet {motor_eff:.1%} er lavere end typisk")
    
    if radius > 100:
        warnings.append(f"⚠️ Meget stor radius - kommerciel mølle?")
    
    return warnings

def design_turbine(target_power_w, wind_speed, cp_target=0.45, motor_eff=0.8):
    """Design mølle omvendt: hvad skal radius være for denne effekt?"""
    # P = 0.5 * ρ * π * r² * v³ * Cp * η
    # r² = P / (0.5 * ρ * π * v³ * Cp * η)
    
    rho = 1.225
    denominator = 0.5 * rho * PI * (wind_speed ** 3) * cp_target * motor_eff
    
    if denominator <= 0:
        return None
    
    r_squared = target_power_w / denominator
    if r_squared < 0:
        return None
    
    return np.sqrt(r_squared)

def sensitivity_analysis(base_power, base_radius, base_tsr, base_cp):
    """Analyse hvor følsom systemet er overfor ændringer"""
    # Test ±10% variation
    variations = {}
    
    # Radius +10%
    p_r_plus = 0.5 * 1.225 * PI * (base_radius * 1.1)**2 * 5**3 * base_cp
    variations['Radius +10%'] = ((p_r_plus - base_power) / base_power) * 100 if base_power != 0 else 0
    
    # Cp +10%
    p_cp_plus = 0.5 * 1.225 * PI * base_radius**2 * 5**3 * (base_cp * 1.1)
    variations['Cp +10%'] = ((p_cp_plus - base_power) / base_power) * 100 if base_power != 0 else 0
    
    # Wind +10%
    p_v_plus = 0.5 * 1.225 * PI * base_radius**2 * (5 * 1.1)**3 * base_cp
    variations['Vind +10%'] = ((p_v_plus - base_power) / base_power) * 100 if base_power != 0 else 0
    
    return variations

def generate_pdf_report(filename, turbine_data, results_df):
    """Generer PDF rapport"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
    )
    
    elements.append(Paragraph("Vindmølle Analyse Rapport", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Mølle data
    turbine_info = [
        ['Parameter', 'Værdi'],
        ['Radius', f"{turbine_data.get('radius', 0):.2f} m"],
        ['Cp værdi', f"{turbine_data.get('cp', 0):.3f}"],
        ['Motor effektivitet', f"{turbine_data.get('efficiency', 0):.1%}"],
        ['Genereret dato', datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    
    turbine_table = Table(turbine_info, colWidths=[3*inch, 2*inch])
    turbine_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(turbine_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Results table
    results_data = [results_df.columns.tolist()] + results_df.values.tolist()
    results_table = Table(results_data, colWidths=[0.8*inch]*len(results_df.columns))
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(results_table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Beregningsfunktioner
def calculate_rpm_from_wind_speed(v_wind, radius, tip_speed_ratio):
    """Beregn RPM fra vindhastighed"""
    if v_wind == 0:
        return 0
    return (tip_speed_ratio * v_wind) / radius * 60 / (2 * PI)

def calculate_mechanical_power(v_wind, radius, rho, cp):
    """Beregn mekanisk effekt: P = 0.5 * ρ * π * r² * v³ * Cp"""
    return 0.5 * rho * PI * radius**2 * v_wind**3 * cp

def calculate_rms_voltage(rpm, kv_value, gear_ratio_val):
    """Beregn RMS spænding fra RPM"""
    return (rpm * gear_ratio_val * kv_value) / np.sqrt(2)

def calculate_electrical_power(p_mech, efficiency):
    """Beregn elektrisk effekt"""
    return p_mech * efficiency

def calculate_rms_current(p_elec, v_rms):
    """Beregn RMS strøm"""
    if v_rms == 0:
        return 0
    return p_elec / v_rms

def calculate_power_after_diodes(v_rms, v_drop_val, i_rms):
    """Beregn effekt efter dioder: (V - V_drop) * I"""
    v_effective = max(0, v_rms - v_drop_val)
    return v_effective * i_rms

def calculate_loss_percentage(p_mech, p_elec_before_diodes, p_after_diodes):
    """Beregn energitab i procent"""
    if p_mech == 0:
        return 0, 0
    motor_loss = ((p_mech - p_elec_before_diodes) / p_mech) * 100
    diode_loss = ((p_elec_before_diodes - p_after_diodes) / p_mech) * 100 if p_elec_before_diodes > 0 else 0
    return motor_loss, diode_loss

# FYSIK BEREGNINGSFUNKTIONER
def calculate_torque(power, rpm):
    """Beregn moment/torque: M = P / ω, hvor ω = RPM * 2π / 60"""
    if rpm == 0:
        return 0
    omega = rpm * 2 * PI / 60
    return power / omega if omega > 0 else 0

def calculate_angular_velocity(rpm):
    """Konverter RPM til rad/s: ω = RPM * 2π / 60"""
    return rpm * 2 * PI / 60

def calculate_reynolds_number(velocity, characteristic_length, kinematic_viscosity=1.81e-5):
    """Beregn Reynolds tal: Re = ρ * v * L / μ"""
    # kinematic_viscosity for luft ved 15°C ≈ 1.81e-5 m²/s
    if characteristic_length == 0:
        return 0
    return (velocity * characteristic_length) / kinematic_viscosity

def calculate_centrifugal_force(mass, rpm, radius_mass):
    """Beregn centrifugal kraft: F = m * ω² * r"""
    omega = rpm * 2 * PI / 60
    return mass * (omega**2) * radius_mass

def estimate_noise_level(rpm):
    """Estimér støjniveau i dB baseret på RPM"""
    # Grundlæggende model: dB øges med RPM
    # ~40 dB ved 0 RPM, stiger ~0.02 dB per RPM
    base_noise = 40
    rpm_factor = rpm * 0.02
    return base_noise + rpm_factor

def calculate_wind_shear(v_ref, z_ref, z_current, shear_exponent=0.2):
    """Beregn vind shear effekt: v = v_ref * (z/z_ref)^α"""
    if z_ref == 0:
        return v_ref
    return v_ref * ((z_current / z_ref) ** shear_exponent)

def calculate_yaw_error_penalty(yaw_angle_deg):
    """Beregn tab fra yaw error (vinkel mellem mølle og vindretning)"""
    # Approksimation: tab = cos(angle)^3.5 for små vinkler
    # Returnerer faktor mellem 0 og 1 (ikke procent)
    yaw_rad = np.radians(yaw_angle_deg)
    penalty = np.cos(yaw_rad) ** 3.5
    return max(0, min(1.0, penalty))

def calculate_pitch_angle_effect(pitch_angle_deg):
    """Optimal pitch angle effekt på Cp"""
    # Approksimation: Cp maksimum ved pitch ≈ -2.5°
    # Cp falder når pitch afviger fra optimum
    optimal_pitch = -2.5
    deviation = abs(pitch_angle_deg - optimal_pitch)
    cp_factor = 1 - (deviation / 100)  # Faktor fra 0-1
    return max(0, cp_factor)

def calculate_tip_vortex_efficiency(tip_speed_ratio):
    """Effektivitet fra tip vortex induktion baseret på TSR"""
    # Ved optimalt TSR (~7) er effektiviteten bedst
    # Faktor omkring 0.95-1.0
    optimal_tsr = 7.0
    if tip_speed_ratio <= 0:
        return 0.8
    efficiency = 1 - (abs(tip_speed_ratio - optimal_tsr) / 50)
    return max(0.7, min(1.0, efficiency))

def calculate_air_density_from_altitude(altitude_m, sea_level_rho=1.225):
    """Beregn lufttæthed baseret på højde"""
    # Barometrisk formel: ρ = ρ0 * exp(-g*M*h / (R*T))
    # Approksimation: ρ ≈ ρ0 * (1 - 0.0065*h / T0)^5.255
    temp_0 = 288.15  # K (15°C)
    lapse_rate = 0.0065
    exponent = 5.255
    return sea_level_rho * ((1 - lapse_rate * altitude_m / temp_0) ** exponent)

def calculate_solidity_ratio(blade_chord, num_blades, rotor_radius):
    """Beregn solidity ratio: σ = (c * B) / (π * R)"""
    if rotor_radius == 0:
        return 0
    return (blade_chord * num_blades) / (PI * rotor_radius)

def calculate_weibull_aep(avg_speed, k_shape=2.0, ref_power_curve=None):
    """Estimér årlig energiproduktion med Weibull distribution"""
    # Weibull distribution for vind
    # AEP ≈ 8760 timer * avg_power beregnet med fordelingen
    # Enkel approksimation: skalering med k-parameter
    hours_per_year = 365.25 * 24
    # For k=2 (Rayleigh), multiplicerer vi med ~0.9 factor
    weibull_factor = 1 - (k_shape - 2) * 0.05
    return hours_per_year * weibull_factor

# Beregn værdier for vindhastigheder
wind_speeds = np.arange(v_min, v_max + 0.1, 1.0)

# Beregn mekanisk effekt for alle typer
p_mech_betz = [calculate_mechanical_power(v, radius, rho, cp_betz) for v in wind_speeds]
p_mech_modern = [calculate_mechanical_power(v, radius, rho, cp_modern) for v in wind_speeds]
p_mech_hobby = [calculate_mechanical_power(v, radius, rho, cp_hobby) for v in wind_speeds]

# Beregn elektrisk effekt før dioder
p_elec_before_diodes_betz = [calculate_electrical_power(p, motor_efficiency) for p in p_mech_betz]
p_elec_before_diodes_modern = [calculate_electrical_power(p, motor_efficiency) for p in p_mech_modern]
p_elec_before_diodes_hobby = [calculate_electrical_power(p, motor_efficiency) for p in p_mech_hobby]

# Beregn RPM
rpm_data_betz = [calculate_rpm_from_wind_speed(v, radius, tip_speed_ratio) for v in wind_speeds]

# Beregn RMS spænding for betz (bruges til at beregne strøm)
v_rms_data_betz = [calculate_rms_voltage(rpm, kv, gear_ratio) for rpm in rpm_data_betz]

# Beregn RMS strøm
i_rms_data_betz = [calculate_rms_current(p_elec, v_rms) for p_elec, v_rms in zip(p_elec_before_diodes_betz, v_rms_data_betz)]

# Beregn effekt efter dioder
p_after_diodes_betz = [calculate_power_after_diodes(v_rms, v_drop, i_rms) 
                       for v_rms, i_rms in zip(v_rms_data_betz, i_rms_data_betz)]

# Vælg mølletype til detaljeret analyse
st.sidebar.subheader("Mølletype")
turbine_type = st.sidebar.radio(
    "Vælg mølletype til detaljeret analyse:",
    ["Moderne vindmølle", "Lille hobbymølle"],
    index=0,
    key="turbine_type"
)

# Gem valgt type
if turbine_type == "Moderne vindmølle":
    p_mech_selected = p_mech_modern
    p_elec_before_selected = p_elec_before_diodes_modern
    cp_selected = cp_modern
    turbine_name = "Moderne"
else:
    p_mech_selected = p_mech_hobby
    p_elec_before_selected = p_elec_before_diodes_hobby
    cp_selected = cp_hobby
    turbine_name = "Hobby"

# Beregn udgangdata for valgt mølle
rpm_data = [calculate_rpm_from_wind_speed(v, radius, tip_speed_ratio) for v in wind_speeds]
v_rms_data = [calculate_rms_voltage(rpm, kv, gear_ratio) for rpm in rpm_data]
i_rms_data = [calculate_rms_current(p_elec, v_rms) for p_elec, v_rms in zip(p_elec_before_selected, v_rms_data)]
p_diodes_data = [calculate_power_after_diodes(v_rms, v_drop, i_rms) 
                 for v_rms, i_rms in zip(v_rms_data, i_rms_data)]

# Beregn energitab
motor_losses = []
diode_losses = []
for i, v in enumerate(wind_speeds):
    # Beregn effekt før dioder
    p_before = p_elec_before_selected[i]
    p_after = p_diodes_data[i]
    m_loss, d_loss = calculate_loss_percentage(p_mech_selected[i], p_before, p_after)
    motor_losses.append(m_loss)
    diode_losses.append(d_loss)

# BEREGN FYSIK PARAMETRE
# Torque beregninger
torques_selected = [calculate_torque(p, rpm) for p, rpm in zip(p_mech_selected, rpm_data)]

# Reynolds tal (karakteristisk længde = blade chord)
reynolds_numbers = [calculate_reynolds_number(v, blade_chord) for v in wind_speeds]

# Centrifugal kraft
centrifugal_forces = [calculate_centrifugal_force(blade_mass, rpm, blade_cm_radius) 
                      for rpm in rpm_data]

# Støj estimat
noise_levels = [estimate_noise_level(rpm) for rpm in rpm_data]

# Wind shear effekt (beregn øget vind ved installationshøjde)
reference_height = 10  # Reference 10m
wind_speeds_at_height = [calculate_wind_shear(v, reference_height, installation_height) 
                         for v in wind_speeds]

# Yaw error penalty
yaw_penalty_factor = calculate_yaw_error_penalty(yaw_angle)

# Pitch angle effekt
pitch_efficiency_factor = calculate_pitch_angle_effect(pitch_angle)

# Tip vortex effektivitet
tip_vortex_eff = calculate_tip_vortex_efficiency(tip_speed_ratio)

# Lufttæthed ved højde
rho_at_height = calculate_air_density_from_altitude(installation_height, rho)

# Solidity ratio
solidity = calculate_solidity_ratio(blade_chord, num_blades, radius)

# Angular velocity
angular_velocities = [calculate_angular_velocity(rpm) for rpm in rpm_data]

# INPUT VALIDERING
st.markdown("---")
st.subheader("Input Validering")
warnings = validate_inputs(radius, cp_selected, tip_speed_ratio, motor_efficiency)
if warnings:
    for warning in warnings:
        st.warning(warning)
else:
    st.success("✓ Alle inputs virker realistiske!")

# KOMMERCIELLE MØLLER SAMMENLIGNING
st.markdown("---")
st.subheader("Sammenlign med Kommercielle Møller")
with st.expander("Sammenlign din mølle med kendt møller"):
    selected_commercial = st.selectbox(
        "Vælg kommerciel mølle:",
        list(commercial_turbines.keys()),
        index=list(commercial_turbines.keys()).index(st.session_state.selected_commercial)
        if st.session_state.selected_commercial in commercial_turbines
        else 0,
        key="selected_commercial"
    )
    
    if selected_commercial:
        comm_data = commercial_turbines[selected_commercial]
        
        # Beregn effekt for begge møller ved samme vind
        test_wind = 10  # m/s
        custom_power = 0.5 * rho * PI * radius**2 * test_wind**3 * cp_selected
        comm_power = 0.5 * rho * PI * comm_data['radius']**2 * test_wind**3 * comm_data['cp']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Din mølle effekt", f"{custom_power:.0f} W", f"ved {test_wind} m/s")
        with col2:
            st.metric(f"{selected_commercial} effekt", f"{comm_power:.0f} W", f"ved {test_wind} m/s")
        
        comparison_table = pd.DataFrame({
            'Parameter': ['Radius', 'Cp', 'Effekt @ 10m/s', 'Nominelt (hvis kendt)'],
            'Din Mølle': [
                f"{radius:.2f} m",
                f"{cp_selected:.3f}",
                f"{custom_power:.0f} W",
                "N/A"
            ],
            selected_commercial: [
                f"{comm_data['radius']:.2f} m",
                f"{comm_data['cp']:.3f}",
                f"{comm_power:.0f} W",
                f"{comm_data['power_rated']} W"
            ]
        })
        
        st.dataframe(comparison_table, use_container_width=True)

# Layout: To kolonner for grafer
st.subheader("Interaktive Grafer (hover for værdier)")
col1, col2 = st.columns(2)

# Graf 1: Mekanisk effekt sammenligning - PLOTLY INTERAKTIV
with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=wind_speeds, y=p_mech_betz, mode='lines+markers', name='Betz (59.3%)', marker=dict(size=8)))
    fig1.add_trace(go.Scatter(x=wind_speeds, y=p_mech_modern, mode='lines+markers', name='Moderne (45%)', marker=dict(size=8)))
    fig1.add_trace(go.Scatter(x=wind_speeds, y=p_mech_hobby, mode='lines+markers', name='Hobby (20%)', marker=dict(size=8)))
    fig1.update_layout(title='Teoretisk Mekanisk Effekt', xaxis_title='Vindhastighed [m/s]', 
                      yaxis_title='Effekt [W]', hovermode='x unified', height=500)
    if use_log_scale:
        fig1.update_yaxes(type="log")
    st.plotly_chart(fig1, use_container_width=True)

# Graf 2: Elektrisk effekt sammenligning - PLOTLY INTERAKTIV
with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=wind_speeds, y=p_elec_before_diodes_betz, mode='lines+markers', name='Betz (59.3%)', marker=dict(size=8)))
    fig2.add_trace(go.Scatter(x=wind_speeds, y=p_elec_before_diodes_modern, mode='lines+markers', name='Moderne (45%)', marker=dict(size=8)))
    fig2.add_trace(go.Scatter(x=wind_speeds, y=p_elec_before_diodes_hobby, mode='lines+markers', name='Hobby (20%)', marker=dict(size=8)))
    fig2.update_layout(title='Elektrisk Effekt (efter motor)', xaxis_title='Vindhastighed [m/s]', 
                      yaxis_title='Effekt [W]', hovermode='x unified', height=500)
    if use_log_scale:
        fig2.update_yaxes(type="log")
    st.plotly_chart(fig2, use_container_width=True)

# Tab-visualisering
st.subheader("Energitab Analyse")
col1, col2 = st.columns(2)

with col1:
    fig_loss1 = go.Figure()
    fig_loss1.add_trace(go.Bar(
        x=wind_speeds,
        y=motor_losses,
        name='Motor tab (%)',
        hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Motor tab</b>: %{y:.2f}%<extra></extra>'
    ))
    fig_loss1.add_trace(go.Bar(
        x=wind_speeds,
        y=diode_losses,
        name='Diode tab (%)',
        hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Diode tab</b>: %{y:.2f}%<extra></extra>'
    ))
    fig_loss1.update_layout(
        title=f'Energitab for {turbine_name} mølle',
        xaxis_title='Vindhastighed [m/s]',
        yaxis_title='Energitab [%]',
        height=500,
        barmode='group',
        hovermode='x unified'
    )
    st.plotly_chart(fig_loss1, use_container_width=True)

with col2:
    # Effektivitet over tid
    efficiency_total = []
    for p_mech, p_after in zip(p_mech_selected, p_diodes_data):
        if p_mech > 0:
            efficiency_total.append((p_after / p_mech) * 100)
        else:
            efficiency_total.append(0)
    
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=wind_speeds,
        y=efficiency_total,
        mode='lines+markers',
        name='Effektivitet',
        fill='tozeroy',
        line=dict(color='green', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Effektivitet</b>: %{y:.2f}%<extra></extra>'
    ))
    fig_eff.update_layout(
        title=f'Samlet Effektivitet for {turbine_name} mølle',
        xaxis_title='Vindhastighed [m/s]',
        yaxis_title='Samlet Effektivitet [%]',
        height=500,
        yaxis=dict(range=[0, 100]),
        hovermode='x unified'
    )
    st.plotly_chart(fig_eff, use_container_width=True)

# Mølle-sammenligning
if compare_turbines:
    st.subheader("Side-by-Side Mølle Sammenligning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compare_type1 = st.selectbox("Mølle 1:", ["Betz grænse", "Moderne", "Hobby"], key="comp1", index=0)
        if compare_type1 == "Betz grænse":
            data1 = p_after_diodes_betz
            name1 = "Betz grænse"
        elif compare_type1 == "Moderne":
            data1 = p_elec_before_diodes_modern
            name1 = "Moderne"
        else:
            data1 = p_elec_before_diodes_hobby
            name1 = "Hobby"
    
    with col2:
        compare_type2 = st.selectbox("Mølle 2:", ["Betz grænse", "Moderne", "Hobby"], key="comp2", index=1)
        if compare_type2 == "Betz grænse":
            data2 = p_after_diodes_betz
            name2 = "Betz grænse"
        elif compare_type2 == "Moderne":
            data2 = p_elec_before_diodes_modern
            name2 = "Moderne"
        else:
            data2 = p_elec_before_diodes_hobby
            name2 = "Hobby"
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=wind_speeds,
        y=data1,
        mode='lines+markers',
        name=name1,
        line=dict(width=2),
        marker=dict(size=8),
        hovertemplate='<b>%{fullData.name}</b><br>Vindhastighed: %{x:.1f} m/s<br>Effekt: %{y:.1f} W<extra></extra>'
    ))
    fig_comp.add_trace(go.Scatter(
        x=wind_speeds,
        y=data2,
        mode='lines+markers',
        name=name2,
        line=dict(width=2),
        marker=dict(size=8),
        hovertemplate='<b>%{fullData.name}</b><br>Vindhastighed: %{x:.1f} m/s<br>Effekt: %{y:.1f} W<extra></extra>'
    ))
    fig_comp.update_layout(
        title='Sammenligning af Møller',
        xaxis_title='Vindhastighed [m/s]',
        yaxis_title='Effekt efter dioder [W]',
        height=500,
        yaxis_type='log' if use_log_scale else 'linear',
        hovermode='x unified'
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# Graf 3: Udgangdata for valgt mølle
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=wind_speeds,
    y=v_rms_data,
    mode='lines+markers',
    name='RMS Spænding [V]',
    line=dict(color='blue', width=2),
    marker=dict(size=8),
    yaxis='y',
    hovertemplate='<b>Spænding</b><br>Vindhastighed: %{x:.1f} m/s<br>Værdi: %{y:.2f} V<extra></extra>'
))
fig3.add_trace(go.Scatter(
    x=wind_speeds,
    y=i_rms_data,
    mode='lines+markers',
    name='RMS Strøm [A]',
    line=dict(color='orange', width=2),
    marker=dict(size=8),
    yaxis='y2',
    hovertemplate='<b>Strøm</b><br>Vindhastighed: %{x:.1f} m/s<br>Værdi: %{y:.2f} A<extra></extra>'
))
fig3.add_trace(go.Scatter(
    x=wind_speeds,
    y=p_diodes_data,
    mode='lines+markers',
    name='Effekt efter dioder [W]',
    line=dict(color='green', width=2),
    marker=dict(size=8),
    yaxis='y3',
    hovertemplate='<b>Effekt</b><br>Vindhastighed: %{x:.1f} m/s<br>Værdi: %{y:.1f} W<extra></extra>'
))

fig3.update_layout(
    title=f'Udgangdata - {turbine_name}',
    xaxis_title='Vindhastighed [m/s]',
    yaxis=dict(
        title=dict(text='RMS Spænding [V]', font=dict(color='blue')),
        tickfont=dict(color='blue')
    ),
    yaxis2=dict(
        title=dict(text='RMS Strøm [A]', font=dict(color='orange')),
        tickfont=dict(color='orange'),
        anchor='free',
        overlaying='y',
        side='left',
        position=0.13
    ),
    yaxis3=dict(
        title=dict(text='Effekt efter dioder [W]', font=dict(color='green')),
        tickfont=dict(color='green'),
        anchor='x',
        overlaying='y',
        side='right'
    ),
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig3, use_container_width=True)

# Graf 4: Blade RPM og Motor RPM vs Vindhastighed
st.subheader("RPM ved forskellige vindhastigheder")

# Beregn motor RPM (påvirket af gear ratio)
motor_rpm_data = [rpm * gear_ratio for rpm in rpm_data]

fig_rpm = go.Figure()

# Blade RPM
fig_rpm.add_trace(go.Scatter(
    x=wind_speeds,
    y=rpm_data,
    mode='lines+markers',
    name='Blade RPM',
    line=dict(color='teal', width=3),
    marker=dict(size=10, symbol='circle'),
    fill='tozeroy',
    fillcolor='rgba(0, 128, 128, 0.2)',
    hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Blade RPM</b>: %{y:.0f}<extra></extra>'
))

# Motor RPM
fig_rpm.add_trace(go.Scatter(
    x=wind_speeds,
    y=motor_rpm_data,
    mode='lines+markers',
    name=f'Motor RPM (gear: {gear_ratio:.1f}x)',
    line=dict(color='purple', width=3, dash='dash'),
    marker=dict(size=10, symbol='diamond'),
    hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Motor RPM</b>: %{y:.0f}<extra></extra>'
))

# Tilføj referencelinje for typiske RPM grænser
fig_rpm.add_hline(y=500, line_dash='dot', line_color='orange', 
                   annotation_text='500 RPM', annotation_position='right')
fig_rpm.add_hline(y=1000, line_dash='dot', line_color='red', 
                   annotation_text='1000 RPM', annotation_position='right')

fig_rpm.update_layout(
    title=f'Blade og Motor omdrejningshastighed - {turbine_name} mølle',
    xaxis_title='Vindhastighed [m/s]',
    yaxis_title='Omdrejninger per minut [RPM]',
    height=500,
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

st.plotly_chart(fig_rpm, use_container_width=True)

# FYSIK OG FORMLER SEKTION
st.markdown("---")
st.header("Fysik & Formler")

# Formler i expandable sektion
with st.expander("Alle Beregningsformler"):
    st.markdown("""
    ### Effekt Beregninger
    
    **Mekanisk Vindenergi:**
    $$P_{mech} = \\frac{1}{2} \\rho A v^3 C_p$$
    - ρ = lufttæthed [kg/m³]
    - A = rotor area = πr²
    - v = vindhastighed [m/s]
    - Cp = effektkoefficient
    
    **Elektrisk Effekt:**
    $$P_{elec} = P_{mech} \\times \\eta_{motor}$$
    
    **Effekt efter Dioder:**
    $$P_{out} = (V_{rms} - V_{drop}) \\times I_{rms}$$
    
    ---
    
    ### Rotations Parametre
    
    **RPM fra vindhastighed:**
    $$RPM = \\frac{\\lambda v}{r} \\times \\frac{60}{2\\pi}$$
    - λ = Tip Speed Ratio
    
    **Angular Velocity:**
    $$\\omega [rad/s] = RPM \\times \\frac{2\\pi}{60}$$
    
    **Moment/Torque:**
    $$M = \\frac{P}{\\omega}$$
    
    ---
    
    ### Aerodynamik
    
    **Reynolds Tal:**
    $$Re = \\frac{v \\times L}{\\nu}$$
    - L = karakteristisk længde (blade chord)
    - ν = kinematisk viskositet ≈ 1.81×10⁻⁵ m²/s
    
    **Wind Shear Effekt:**
    $$v(z) = v_{ref} \\times \\left(\\frac{z}{z_{ref}}\\right)^\\alpha$$
    - α = shear eksponent ≈ 0.2
    
    **Solidity Ratio:**
    $$\\sigma = \\frac{c \\times B}{\\pi R}$$
    - c = blade chord
    - B = antal blades
    
    ---
    
    ### Mechaniske Kræfter
    
    **Centrifugal Kraft:**
    $$F_c = m \\times \\omega^2 \\times r$$
    - m = blade masse
    - r = massecenters radius
    
    **Yaw Error Penalty:**
    $$Penalty = \\cos(\\theta)^{3.5} \\times 100\\%$$
    - θ = yaw vinkel
    
    **Pitch Angle Effekt:**
    Optimal pitch ≈ -2.5° for maksimal Cp
    
    ---
    
    ### Støj & Øvrige
    
    **Støj Estimat:**
    $$dB ≈ 40 + RPM \\times 0.02$$
    
    **Lufttæthed vs. Højde:**
    $$\\rho(h) = \\rho_0 \\times \\left(1 - \\frac{0.0065h}{T_0}\\right)^{5.255}$$
    """)

# Fysik data tab
st.subheader("Fysik Parametre og Karakteristika")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Solidity Ratio", f"{solidity:.4f}")
    st.metric("Tip Vortex Effektivitet", f"{tip_vortex_eff:.2%}")
    
with col2:
    st.metric("Yaw Penalty", f"{yaw_penalty_factor:.2%}")
    st.metric("Pitch Effektivitet", f"{pitch_efficiency_factor:.2%}")
    
with col3:
    st.metric("Lufttæthed ved højde", f"{rho_at_height:.4f} kg/m³")
    st.metric("TSR (optimal ~7)", f"{tip_speed_ratio:.2f}")
    
with col4:
    st.metric("Installation højde", f"{installation_height:.1f} m")
    st.metric("Yaw Vinkel", f"{yaw_angle}°")

# Fysik grafer
col1, col2 = st.columns(2)

with col1:
    # Torque graf
    fig_torque = go.Figure()
    fig_torque.add_trace(go.Scatter(
        x=wind_speeds,
        y=torques_selected,
        mode='lines+markers',
        name='Torque',
        fill='tozeroy',
        line=dict(color='purple', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Moment</b>: %{y:.2f} Nm<extra></extra>'
    ))
    fig_torque.update_layout(
        title='Moment/Torque på Akslen',
        xaxis_title='Vindhastighed [m/s]',
        yaxis_title='Moment [Nm]',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_torque, use_container_width=True)

with col2:
    # Støj graf
    fig_noise = go.Figure()
    fig_noise.add_trace(go.Scatter(
        x=wind_speeds,
        y=noise_levels,
        mode='lines+markers',
        name='Støjniveau',
        line=dict(color='red', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Støj</b>: %{y:.1f} dB<extra></extra>'
    ))
    fig_noise.add_hline(y=85, line_dash='dash', line_color='orange', annotation_text='Hørskadigraf (85 dB)')
    fig_noise.add_hline(y=70, line_dash='dash', line_color='yellow', annotation_text='Normalt tale niveau (70 dB)')
    fig_noise.update_layout(
        title='Estimeret Støjniveau',
        xaxis_title='Vindhastighed [m/s]',
        yaxis_title='Støjniveau [dB]',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_noise, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    # Reynolds tal
    fig_re = go.Figure()
    fig_re.add_trace(go.Scatter(
        x=wind_speeds,
        y=reynolds_numbers,
        mode='lines+markers',
        name='Reynolds Tal',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Re</b>: %{y:.0f}<extra></extra>'
    ))
    fig_re.add_hline(y=500000, line_dash='dash', line_color='red', annotation_text='Transition område')
    fig_re.update_layout(
        title='Reynolds Tal (Aerodynamisk regime)',
        xaxis_title='Vindhastighed [m/s]',
        yaxis_title='Reynolds Tal',
        yaxis_type='log',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_re, use_container_width=True)

with col2:
    # Centrifugal kraft
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Scatter(
        x=wind_speeds,
        y=centrifugal_forces,
        mode='lines+markers',
        name='Centrifugal Kraft',
        fill='tozeroy',
        line=dict(color='brown', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Vindhastighed</b>: %{x:.1f} m/s<br><b>Kraft</b>: %{y:.2f} N<extra></extra>'
    ))
    fig_cf.update_layout(
        title='Centrifugal Kraft på Blades',
        xaxis_title='Vindhastighed [m/s]',
        yaxis_title='Centrifugal Kraft [N]',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_cf, use_container_width=True)

# Tabel med beregnede værdier
st.subheader(f"Detaljerede Beregninger - {turbine_name}")

results_data = {
    'V [m/s]': [f"{v:.1f}" for v in wind_speeds],
    'RPM': [f"{rpm:.0f}" for rpm in rpm_data],
    'ω [rad/s]': [f"{w:.2f}" for w in angular_velocities],
    'P_mech [W]': [f"{p:.1f}" for p in p_mech_selected],
    'M [Nm]': [f"{t:.3f}" for t in torques_selected],
    'P_elec [W]': [f"{p:.1f}" for p in p_elec_before_selected],
    'dB': [f"{db:.1f}" for db in noise_levels],
    'Re': [f"{re:.0f}" for re in reynolds_numbers],
    'F_c [N]': [f"{f:.2f}" for f in centrifugal_forces],
    'η [%]': [f"{e:.1f}" for e in efficiency_total]
}

results_df = pd.DataFrame(results_data)
st.dataframe(results_df, use_container_width=True)

# Specifik værdi-eksport
st.subheader("Hent Specifik Værdi")
col1, col2 = st.columns(2)

st.session_state.selected_wind = min(
    max(float(st.session_state.selected_wind), float(v_min)),
    float(v_max)
)

with col1:
    selected_wind = st.number_input(
        "Vælg vindhastighed [m/s]",
        min_value=float(v_min),
        max_value=float(v_max),
        value=float(st.session_state.selected_wind),
        step=0.5,
        key="selected_wind"
    )

with col2:
    if st.button("Hent Værdier"):
        # Find nærmeste værdi
        idx = np.argmin(np.abs(wind_speeds - selected_wind))
        closest_wind = wind_speeds[idx]
        
        st.info(f"""
        **Vindhastighed:** {closest_wind:.1f} m/s
        
        **Mekanisk effekt:** {p_mech_selected[idx]:.2f} W
        
        **Elektrisk effekt (før dioder):** {p_elec_before_selected[idx]:.2f} W
        
        **RMS Spænding:** {v_rms_data[idx]:.2f} V
        
        **RMS Strøm:** {i_rms_data[idx]:.3f} A
        
        **Effekt efter dioder:** {p_diodes_data[idx]:.2f} W
        
        **Motor tab:** {motor_losses[idx]:.1f}%
        
        **Diode tab:** {diode_losses[idx]:.1f}%
        
        **Samlet effektivitet:** {efficiency_total[idx]:.1f}%
        """)

# EXPORT FUNKTIONER
st.markdown("---")
st.subheader("Eksporter Resultater")

col1, col2, col3 = st.columns(3)

with col1:
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download som CSV",
        data=csv,
        file_name=f"vindmolle_beregning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col2:
    turbine_info = {
        'radius': radius,
        'cp': cp_selected,
        'efficiency': motor_efficiency,
        'tsr': tip_speed_ratio,
        'kv': kv,
    }
    
    if st.button("Generer PDF Rapport"):
        pdf_buffer = generate_pdf_report("rapport", turbine_info, results_df.head(10))
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

with col3:
    if st.button("Gem Beregning til History"):
        calculation_entry = {
            'timestamp': datetime.now().isoformat(),
            'radius': float(radius),
            'cp': float(cp_selected),
            'motor_efficiency': float(motor_efficiency),
            'tsr': float(tip_speed_ratio),
            'avg_power': float(0.5 * rho * PI * radius**2 * 5**3 * cp_selected),
        }
        st.session_state.calculation_history.append(calculation_entry)
        st.success("Beregning gemt!")

# HISTORY/CACHE VISNING
if st.session_state.calculation_history:
    with st.expander("Se tidligere beregninger"):
        history_df = pd.DataFrame(st.session_state.calculation_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Ryd History"):
            st.session_state.calculation_history = []
            st.rerun()

# Footer med information
st.markdown("---")
st.markdown("""
### Noter:
- Alle beregninger er baseret på ideelle forhold
- Lufttæthed (ρ) sættes til 1.225 kg/m³ som standard (normalt lufttryk ved havniveau)
- RPM beregnes ud fra Tip Speed Ratio og gearudveksling
- Elektrisk effekt inkluderer motor-effektivitet
- Effekt efter dioder reduceres med spændingsfald over dioden
- Økonomisk analyse er baseret på gennemsnitlige vindforholds forudsætninger
""")
