"""
Vindmølle Blade Designer - Fase 1-3
NACA Profil Generator → Blade Geometri → BEM Optimering → Fusion 360 Export
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
import matplotlib.pyplot as plt

# ============================================================================
# FASE 1: NACA PROFIL GENERATOR
# ============================================================================

def naca_4_digit(code, num_points=100, spacing="cosine", closed_te=True):
    """Genererer NACA 4-digit airfoil koordinater"""
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA kode skal være 4 cifre (fx 2412).")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:4]) / 100.0
    
    if spacing == "cosine":
        beta = np.linspace(0, np.pi, num_points)
        x = 0.5 * (1 - np.cos(beta))
    else:
        x = np.linspace(0, 1, num_points)
    
    # Tykkelsesfordeling (thickness distribution)
    te_coeff = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                  0.2843 * x**3 + te_coeff * x**4)
    
    # Camber linje
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if p == 0:
        yc[:] = 0
        dyc_dx[:] = 0
    else:
        for i, xi in enumerate(x):
            if xi < p:
                yc[i] = (m / p**2) * (2 * p * xi - xi**2)
                dyc_dx[i] = (2 * m / p**2) * (p - xi)
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * xi - xi**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - xi)
    
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    xl = x + yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    yl = yc - yt * np.cos(theta)
    
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    
    return np.column_stack([x_coords, y_coords])


def is_valid_naca(code):
    return len(code) == 4 and code.isdigit()


def resample_profile(profile, n_points=120):
    profile = np.array(profile, dtype=float)
    s = np.linspace(0, 1, profile.shape[0])
    s_new = np.linspace(0, 1, n_points)
    x_new = np.interp(s_new, s, profile[:, 0])
    y_new = np.interp(s_new, s, profile[:, 1])
    return np.column_stack([x_new, y_new])


def thickness_ratio(profile):
    prof = np.array(profile, dtype=float)
    min_x = np.min(prof[:, 0])
    max_x = np.max(prof[:, 0])
    chord_len = max_x - min_x
    if chord_len <= 0:
        return 0.0
    thickness = np.max(prof[:, 1]) - np.min(prof[:, 1])
    return thickness / chord_len


def get_airfoil_coeffs(alpha_rad, thickness_ratio_val):
    """Placeholder for airfoil polars; uses Viterna-extrapolated synthetic polar."""
    alpha_deg = np.degrees(alpha_rad)
    base_alpha = np.arange(-20, 21, 1)
    base_cl = 2.0 * np.pi * np.radians(base_alpha)
    base_cd = 0.01 + 0.02 * (np.radians(base_alpha) ** 2) + 0.02 * max(0.0, thickness_ratio_val - 0.12)

    alpha_ext, cl_ext, cd_ext = viterna_extrapolation(base_alpha, base_cl, base_cd)
    cl = np.interp(alpha_deg, alpha_ext, cl_ext)
    cd = np.interp(alpha_deg, alpha_ext, cd_ext)
    return cl, cd


def viterna_extrapolation(alpha_deg, cl_data, cd_data, cd_max=1.8):
    """Extrapolate polar data to +/-180° using Viterna method (placeholder)."""
    alpha = np.asarray(alpha_deg)
    cl = np.asarray(cl_data)
    cd = np.asarray(cd_data)
    if alpha.size != cl.size or alpha.size != cd.size:
        raise ValueError("Polar data length mismatch")

    alpha_ext = np.linspace(-180, 180, 361)
    cl_ext = np.interp(alpha_ext, alpha, cl, left=cl[0], right=cl[-1])
    cd_ext = np.interp(alpha_ext, alpha, cd, left=cd[0], right=cd[-1])

    stall_mask = np.abs(alpha_ext) > 20
    cl_ext[stall_mask] = 0.5 * cd_max * np.sin(2 * np.radians(alpha_ext[stall_mask]))
    cd_ext[stall_mask] = cd_max * (np.sin(np.radians(alpha_ext[stall_mask])) ** 2)

    return alpha_ext, cl_ext, cd_ext


def generate_betz_optimized_blade(naca_code, tip_radius, hub_radius, num_sections,
                                  tsr, blades, alpha_opt_deg, cl_design,
                                  min_chord, max_chord, p_axis=0.25):
    """Generér Betz-optimeret blade geometri (simpel BEM-tilnærmelse)"""
    radii = np.linspace(hub_radius, tip_radius, num_sections)
    profile_2d = naca_4_digit(naca_code, num_points=120)

    chords = []
    twists = []
    sections = []

    for r in radii:
        lambda_r = tsr * (r / tip_radius)
        if lambda_r <= 0:
            phi = np.radians(45)
        else:
            phi = np.arctan(2 / (3 * lambda_r))

        twist = np.degrees(phi) - alpha_opt_deg
        chord = (8 * np.pi * r * (np.sin(phi) ** 2)) / (blades * cl_design * np.cos(phi))

        chord = float(np.clip(chord, min_chord, max_chord))

        scaled_profile = profile_2d * chord
        twist_rad = np.radians(twist)
        rotated_profile = scaled_profile.copy()
        rotated_profile[:, 1] = (
            scaled_profile[:, 1] * np.cos(twist_rad)
            - scaled_profile[:, 0] * np.sin(twist_rad)
        )

        chords.append(chord)
        twists.append(twist)
        sections.append({
            "radius": r,
            "chord": chord,
            "twist": twist,
            "thread": p_axis,
            "offset": 0.0,
            "profile": rotated_profile
        })

    return {
        "naca_code": naca_code,
        "num_sections": num_sections,
        "radii": radii,
        "chords": np.array(chords),
        "twists": np.array(twists),
        "sections": sections,
        "profile_2d": profile_2d,
        "tip_radius": tip_radius,
        "hub_radius": hub_radius,
        "tsr": tsr,
        "blades": blades,
        "alpha_opt_deg": alpha_opt_deg,
        "cl_design": cl_design,
        "p_axis": p_axis
    }


def export_qblade_bld(blade_data, polar_name=None):
    """Eksporterer QBlade v2 .bld format som YBlade kan læse"""
    polar_name = polar_name or f"NACA{blade_data['naca_code']}"
    lines = []
    lines.append("Blade Data")
    lines.append("Generated by Windturbine Blade Designer")
    lines.append("POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] P_AXIS POLAR_FILE")

    for s in blade_data["sections"]:
        pos = s["radius"]
        chord = s["chord"]
        twist = s["twist"]
        offset_x = 0.0
        offset_y = s.get("offset", 0.0)
        p_axis = s.get("thread", blade_data.get("p_axis", 0.25))
        lines.append(f"{pos:.5f} {chord:.5f} {twist:.4f} {offset_x:.5f} {offset_y:.5f} {p_axis:.3f} {polar_name}")

    return "\n".join(lines)


def export_afl(naca_code, num_points=200):
    profile = naca_4_digit(naca_code, num_points=num_points)
    lines = [f"NACA {naca_code}"]
    for x, y in profile:
        lines.append(f"{x:.6f} {y:.6f}")
    return "\n".join(lines)


def generate_blade_geometry(root_profile_2d, tip_profile_2d, num_sections=10, root_radius=0.1,
                           tip_radius=2.0, root_chord=0.3, tip_chord=0.1,
                           root_twist=25, tip_twist=0, profile_name="Custom",
                           auto_spacing=False, chords_override=None, twists_override=None):
    """Genererer 3D blade geometri med twist og taper"""

    if auto_spacing:
        beta = np.linspace(0, np.pi, num_sections)
        radii = root_radius + (tip_radius - root_radius) * 0.5 * (1 - np.cos(beta))
    else:
        radii = np.linspace(root_radius, tip_radius, num_sections)
    if chords_override is not None:
        chords = np.array(chords_override, dtype=float)
    else:
        chords = np.linspace(root_chord, tip_chord, num_sections)

    if twists_override is not None:
        twists = np.array(twists_override, dtype=float)
    else:
        twists = np.linspace(root_twist, tip_twist, num_sections)
    
    root_prof = resample_profile(root_profile_2d, n_points=160)
    tip_prof = resample_profile(tip_profile_2d, n_points=160)
    # Normalize both profiles to 0..1 chord
    for prof in (root_prof, tip_prof):
        min_x = np.min(prof[:, 0])
        max_x = np.max(prof[:, 0])
        chord_len = max_x - min_x
        if chord_len > 0:
            prof[:, 0] = (prof[:, 0] - min_x) / chord_len
            prof[:, 1] = prof[:, 1] / chord_len

    blade_sections = []
    for i, (r, chord, twist) in enumerate(zip(radii, chords, twists)):
        span_frac = 0 if tip_radius == root_radius else (r - root_radius) / (tip_radius - root_radius)
        blended = (1 - span_frac) * root_prof + span_frac * tip_prof
        scaled_profile = blended * chord
        
        twist_rad = np.radians(twist)
        rotated_profile = scaled_profile.copy()
        x = scaled_profile[:, 0]
        y = scaled_profile[:, 1]
        # Rotate about quarter-chord to avoid bending artifacts
        x0 = 0.25 * chord
        x_shifted = x - x0
        rotated_profile[:, 0] = x_shifted * np.cos(twist_rad) - y * np.sin(twist_rad) + x0
        rotated_profile[:, 1] = x_shifted * np.sin(twist_rad) + y * np.cos(twist_rad)
        
        blade_sections.append({
            'radius': r,
            'chord': chord,
            'twist': twist,
            'profile': rotated_profile
        })
    
    return {
        'naca_code': profile_name,
        'num_sections': num_sections,
        'radii': radii,
        'chords': chords,
        'twists': twists,
        'sections': blade_sections,
        'profile_2d': root_profile_2d
    }


def export_yblade_csv(blade_data):
    """Eksporterer til YBlade_v2 CSV format"""
    rows = []
    
    for section in blade_data['sections']:
        profile = section['profile']
        
        row = {
            'radius_m': f"{section['radius']:.4f}",
            'chord_m': f"{section['chord']:.4f}",
            'twist_deg': f"{section['twist']:.2f}"
        }
        
        for j, (x, y) in enumerate(profile):
            row[f'x{j}'] = f"{x:.6f}"
            row[f'y{j}'] = f"{y:.6f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def export_xyz_simple(blade_data):
    """Simpel XYZ export for Fusion 360"""
    data = []
    
    for section in blade_data['sections']:
        profile = section['profile']
        
        for x, y in profile:
            data.append({
                'Radius_m': f"{section['radius']:.4f}",
                'X_Chord': f"{x:.6f}",
                'Y_Thick': f"{y:.6f}",
                'Twist_deg': f"{section['twist']:.2f}"
            })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def export_fusion_spline_csv(blade_data):
    """Export coordinates for Fusion 360 spline import (sectioned)."""
    rows = []
    for idx, section in enumerate(blade_data['sections']):
        z = section['radius']
        profile = section['profile']
        for x, y in profile:
            rows.append({
                "section": idx,
                "x": f"{x:.6f}",
                "y": f"{y:.6f}",
                "z": f"{z:.6f}"
            })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def parse_afl(content):
    """Parse .afl/.dat content into Nx2 array of floats."""
    lines = content.strip().splitlines()
    points = []
    for line in lines[1:]:
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue
        try:
            x = float(tokens[0])
            y = float(tokens[1])
            points.append([x, y])
        except ValueError:
            continue
    if not points:
        raise ValueError("Filen indeholder ingen gyldige koordinater.")
    return np.array(points)


def render_blade_3d(blade_data, title="3D Blade Preview"):
    fig = go.Figure()
    for section in blade_data['sections']:
        profile = section['profile']
        fig.add_trace(go.Scatter3d(
            x=profile[:, 0],
            y=profile[:, 1],
            z=np.full(profile.shape[0], section['radius']),
            mode='lines',
            line=dict(width=3)
        ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Chord X',
            yaxis_title='Thick Y',
            zaxis_title='Radius Z'
        ),
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# FASE 2: SIMPLE BEM BEREGNINGER
# ============================================================================

def bem_analysis(blade_data, wind_speed=10, rpm=100, rho=1.225, tsr=None, tol=1e-5, max_iter=200, blades=3):
    """
    Blade Element Momentum teori - simpel version
    Beregner kraft og effekt på hver blade sektion
    """
    
    if tsr is None:
        avg_radius = np.mean(blade_data['radii'])
        omega = rpm * 2 * np.pi / 60
        tsr = (omega * avg_radius) / wind_speed
    
    results = []
    
    omega = rpm * 2 * np.pi / 60
    R = blade_data['radii'][-1]
    r_hub = blade_data['radii'][0]
    dr = blade_data['radii'][1] - blade_data['radii'][0] if len(blade_data['radii']) > 1 else 0.1
    B = blades
    
    for section in blade_data['sections']:
        r = section['radius']
        chord = section['chord']
        twist = np.radians(section['twist'])

        # Iterative induction
        a = 0.3
        a_prime = 0.0
        iter_count = 0
        while iter_count < max_iter:
            v_axial = wind_speed * (1 - a)
            v_tan = omega * r * (1 + a_prime)
            phi = np.arctan2(v_axial, v_tan)
            
            # Tip & hub loss
            sin_phi = np.sin(phi)
            sin_phi = max(1e-6, sin_phi)
            f_tip = (B / 2) * (R - r) / (r * sin_phi)
            f_hub = (B / 2) * (r - r_hub) / (r_hub * sin_phi)
            F_tip = (2 / np.pi) * np.arccos(np.exp(-f_tip))
            F_hub = (2 / np.pi) * np.arccos(np.exp(-f_hub))
            F = np.clip(F_tip * F_hub, 1e-3, 1.0)

            alpha = phi - twist
            t_ratio = thickness_ratio(section['profile'])
            cl, cd = get_airfoil_coeffs(alpha, t_ratio)
            
            cn = cl * np.cos(phi) + cd * np.sin(phi)
            ct = cl * np.sin(phi) - cd * np.cos(phi)
            
            sigma = B * chord / (2 * np.pi * r)

            a_new = 1 / (1 + (4 * F * sin_phi**2) / (sigma * cn + 1e-9))
            # Buhl/Glauert correction for high induction
            if a_new > 0.4:
                ct_mom = 4 * F * a_new * (1 - a_new)
                a_new = 0.5 * (2 + ct_mom - np.sqrt((ct_mom - 2) ** 2 + 4 * (ct_mom - 1)))
                a_new = np.clip(a_new, 0.0, 0.95)

            a_prime_new = 1 / ((4 * F * sin_phi * np.cos(phi)) / (sigma * ct + 1e-9) - 1)

            if abs(a_new - a) < tol and abs(a_prime_new - a_prime) < tol:
                a, a_prime = a_new, a_prime_new
                break
            a, a_prime = a_new, a_prime_new
            iter_count += 1

        v_rel = np.sqrt(v_axial**2 + v_tan**2)
        lift = 0.5 * rho * v_rel**2 * chord * cl
        drag = 0.5 * rho * v_rel**2 * chord * cd
        fn = lift * np.cos(phi) + drag * np.sin(phi)
        ft = lift * np.sin(phi) - drag * np.cos(phi)
        torque = ft * r * dr * B
        power = torque * omega

        results.append({
            'radius': r,
            'twist': np.degrees(twist),
            'chord': chord,
            'v_rel': v_rel,
            'alpha': np.degrees(alpha),
            'cl': cl,
            'cd': cd,
            'lift': lift,
            'drag': drag,
            'fn': fn,
            'ft': ft,
            'torque_section': torque,
            'power_section': power,
            'a': a,
            'a_prime': a_prime,
            'F': F
        })
    
    total_torque = sum([r['torque_section'] for r in results])
    total_power = sum([r['power_section'] for r in results])
    
    return {
        'sections': results,
        'total_torque': total_torque,
        'total_power': total_power,
        'wind_speed': wind_speed,
        'rpm': rpm,
        'tsr': tsr,
        'cp': total_power / (0.5 * rho * np.pi * blade_data['radii'][-1]**2 * wind_speed**3 + 0.001)
    }


# ============================================================================
# FASE 3: OPTIMERING
# ============================================================================

def optimize_blade_for_betz(blade_data, wind_speed=10, target_tsr=7.0):
    """
    Optimerer blade geometri mod Betz grænse (59.3% Cp)
    ved at variere twist distribution
    """
    
    best_cp = 0
    best_rpm = 0
    best_results = None
    
    for rpm in range(50, 300, 10):
        results = bem_analysis(blade_data, wind_speed=wind_speed, rpm=rpm)
        
        if results['cp'] > best_cp:
            best_cp = results['cp']
            best_rpm = rpm
            best_results = results
    
    return {
        'best_cp': best_cp,
        'best_rpm': best_rpm,
        'results': best_results,
        'efficiency': min(best_cp / 0.593, 1.0) * 100  # % of Betz
    }


def schmitz_optimize_geometry(root_profile, tip_profile, hub_radius, tip_radius, num_sections,
                              target_tsr, alpha_opt_deg, cl_design, cd_design, blades,
                              root_chord, tip_chord, root_twist, tip_twist):
    """Schmitz-based chord/twist optimization (simplified)."""
    radii = np.linspace(hub_radius, tip_radius, num_sections)
    chords = []
    twists = []

    for r in radii:
        lam_r = target_tsr * (r / tip_radius)
        if lam_r <= 1e-6:
            phi = np.radians(45)
        else:
            phi = np.arctan(2 / (3 * lam_r))

        # Include L/D effect (placeholder)
        ld = cl_design / max(cd_design, 1e-4)
        phi = np.arctan(np.tan(phi) / (1 + 1 / max(ld, 1e-3)))

        twist = np.degrees(phi) - alpha_opt_deg
        chord = (8 * np.pi * r * (np.sin(phi) ** 2)) / (blades * cl_design * np.cos(phi))

        chords.append(chord)
        twists.append(twist)

    chords = np.clip(chords, min(root_chord, tip_chord), max(root_chord, tip_chord))
    twists = np.clip(twists, min(root_twist, tip_twist), max(root_twist, tip_twist))

    return generate_blade_geometry(
        root_profile_2d=root_profile,
        tip_profile_2d=tip_profile,
        num_sections=num_sections,
        root_radius=hub_radius,
        tip_radius=tip_radius,
        root_chord=chords[0],
        tip_chord=chords[-1],
        root_twist=twists[0],
        tip_twist=twists[-1],
        profile_name="Schmitz",
        auto_spacing=True,
        chords_override=chords,
        twists_override=twists
    )


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Blade Designer", layout="wide")
st.title("🌪️ Vindmølle Blade Designer - Fase 1-3")

# Session state defaults
if "naca_code" not in st.session_state:
    st.session_state.naca_code = "2412"
if "airfoil_source" not in st.session_state:
    st.session_state.airfoil_source = "NACA 4-cifret"
if "airfoil_profile" not in st.session_state:
    st.session_state.airfoil_profile = None
if "airfoil_name" not in st.session_state:
    st.session_state.airfoil_name = None
if "naca_spacing" not in st.session_state:
    st.session_state.naca_spacing = "cosine"
if "naca_closed_te" not in st.session_state:
    st.session_state.naca_closed_te = True
if "geom_spacing" not in st.session_state:
    st.session_state.geom_spacing = st.session_state.naca_spacing
if "geom_closed_te" not in st.session_state:
    st.session_state.geom_closed_te = st.session_state.naca_closed_te
if "blade_data" not in st.session_state:
    st.session_state.blade_data = None

# Sidebar navigation
st.sidebar.markdown("### Navigation")
phase = st.sidebar.radio(
    "Vælg fase:",
    ["Fase 1: NACA Generator", "Fase 2: Geometri Editor", "Fase 3: Optimering"]
)

# ============================================================================
# FASE 1: NACA GENERATOR
# ============================================================================

if phase == "Fase 1: NACA Generator":
    st.header("Fase 1: NACA Profil Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NACA Profil Vælger")
        st.write("NACA 4-digit format: **MPPT** (f.eks. 2412)")
        st.write("- **M**: Max camber (%) - første ciffer")
        st.write("- **P**: Camber position (x/10) - andet ciffer")
        st.write("- **PT**: Tykkelse (%) - sidste to cifre")
        
        airfoil_mode = st.radio(
            "Vælg profilkilde:",
            ["NACA 4-cifret", "Upload .afl/.dat"],
            horizontal=True,
            index=0 if st.session_state.airfoil_source == "NACA 4-cifret" else 1,
            key="phase1_airfoil_source"
        )
        naca_input = st.text_input("Indtast NACA kode:", st.session_state.naca_code, max_chars=4, key="phase1_naca")
        uploaded_airfoil = st.file_uploader(
            "Upload airfoil (.afl/.dat)",
            type=["afl", "dat", "txt"],
            accept_multiple_files=False,
            key="phase1_airfoil_upload"
        )
        
        # Valider input
        if airfoil_mode == "NACA 4-cifret":
            if len(naca_input) == 4 and naca_input.isdigit():
                num_points = st.slider("Antal punkter langs profil:", 50, 200, 100)
                spacing = st.selectbox(
                    "Punktfordeling",
                    ["cosine", "uniform"],
                    index=0 if st.session_state.naca_spacing == "cosine" else 1,
                    key="phase1_spacing"
                )
                closed_te = st.checkbox("Lukket trailing edge", value=bool(st.session_state.naca_closed_te), key="phase1_closed_te")
                profile = naca_4_digit(naca_input, num_points, spacing=spacing, closed_te=closed_te)
                profile_name = f"NACA {naca_input}"
                st.session_state.naca_code = naca_input
                st.session_state.naca_spacing = spacing
                st.session_state.naca_closed_te = closed_te
                st.session_state.airfoil_source = "NACA 4-cifret"
                st.session_state.airfoil_profile = profile
                st.session_state.airfoil_name = profile_name
            else:
                st.error("❌ Indtast gyldig NACA 4-digit kode (f.eks. 2412)")
                profile = None
                profile_name = None
        else:
            if uploaded_airfoil is not None:
                try:
                    content = uploaded_airfoil.getvalue().decode("utf-8", errors="ignore")
                    profile = parse_afl(content)
                    profile_name = uploaded_airfoil.name
                    st.session_state.airfoil_source = "Upload .afl/.dat"
                    st.session_state.airfoil_profile = profile
                    st.session_state.airfoil_name = profile_name
                except Exception as exc:
                    st.error(f"❌ Kunne ikke læse filen: {exc}")
                    profile = None
                    profile_name = None
            else:
                profile = None
                profile_name = None

        if profile is not None:
            
            # Plot med Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=profile[:, 0],
                y=profile[:, 1],
                fill='toself',
                name='Airfoil Profil',
                mode='lines',
                line=dict(color='blue', width=2),
                hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'{profile_name} Airfoil Profil',
                xaxis_title='Chord Position (x/c)',
                yaxis_title='Tykkelse (y/c)',
                height=500,
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Profil statistik
            col1a, col1b, col1c = st.columns(3)
            if airfoil_mode == "NACA 4-cifret":
                col1a.metric("Max Camber", f"{int(naca_input[0])}%")
                col1b.metric("Camber Position", f"{int(naca_input[1])} x 10%")
                col1c.metric("Tykkelse", f"{naca_input[2:4]}%")
            else:
                col1a.metric("Punkter", f"{profile.shape[0]}")
                col1b.metric("Min x", f"{profile[:, 0].min():.3f}")
                col1c.metric("Max x", f"{profile[:, 0].max():.3f}")

# ============================================================================
# FASE 2: GEOMETRI EDITOR
# ============================================================================

elif phase == "Fase 2: Geometri Editor":
    st.header("Fase 2: Blade Geometri Designer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔧 Profil Indstillinger")
        st.caption("Root airfoil")
        root_mode = st.radio(
            "Profilkilde (root):",
            ["NACA 4-cifret", "Upload .afl/.dat"],
            horizontal=True,
            key="phase2_root_airfoil_source"
        )
        root_naca = st.text_input("NACA Kode (root):", st.session_state.naca_code, max_chars=4, key="phase2_root_naca")
        root_upload = st.file_uploader(
            "Upload airfoil (.afl/.dat) (root)",
            type=["afl", "dat", "txt"],
            accept_multiple_files=False,
            key="phase2_root_airfoil_upload"
        )
        if root_mode == "NACA 4-cifret":
            st.selectbox(
                "Punktfordeling (root)",
                ["cosine", "uniform"],
                index=0 if st.session_state.naca_spacing == "cosine" else 1,
                key="geom_spacing"
            )
            st.checkbox("Lukket trailing edge (root)", value=bool(st.session_state.naca_closed_te), key="geom_closed_te")
            st.session_state.naca_spacing = st.session_state.geom_spacing
            st.session_state.naca_closed_te = st.session_state.geom_closed_te

        st.divider()
        same_as_root = st.checkbox("Tip airfoil = root airfoil", value=True, key="phase2_same_airfoil")
        if not same_as_root:
            st.caption("Tip airfoil")
            tip_mode = st.radio(
                "Profilkilde (tip):",
                ["NACA 4-cifret", "Upload .afl/.dat"],
                horizontal=True,
                key="phase2_tip_airfoil_source"
            )
            tip_naca = st.text_input("NACA Kode (tip):", st.session_state.naca_code, max_chars=4, key="phase2_tip_naca")
            tip_upload = st.file_uploader(
                "Upload airfoil (.afl/.dat) (tip)",
                type=["afl", "dat", "txt"],
                accept_multiple_files=False,
                key="phase2_tip_airfoil_upload"
            )
        else:
            tip_mode = root_mode
            tip_naca = root_naca
            tip_upload = root_upload
        num_sections = st.slider("Antal radiale sektioner:", 5, 25, 12)
    
    with col2:
        st.subheader("📏 Blade Dimensioner")
        root_radius = st.number_input("Rod radius [m]:", 0.05, 1.0, 0.15, 0.05)
        tip_radius = st.number_input("Tip radius [m]:", 0.5, 5.0, 2.0, 0.1)
        auto_spacing = st.checkbox("Auto spacing", value=True)
    
    with col3:
        st.subheader("✈️ Chord & Twist")
        root_chord = st.number_input("Rod chord [m]:", 0.1, 1.0, 0.4, 0.05)
        tip_chord = st.number_input("Tip chord [m]:", 0.01, 0.5, 0.05, 0.01)
        root_twist = st.number_input("Rod twist [°]:", 0, 45, 25, 1)
        tip_twist = st.number_input("Tip twist [°]:", -10, 10, 0, 1)
    
    if st.button("🔨 Generer Blade Geometri", key="gen_blade"):
        root_profile = None
        tip_profile = None
        profile_name = None

        if root_mode == "NACA 4-cifret":
            if is_valid_naca(root_naca):
                spacing = st.session_state.geom_spacing
                closed_te = st.session_state.geom_closed_te
                root_profile = naca_4_digit(root_naca, num_points=120, spacing=spacing, closed_te=closed_te)
            else:
                st.error("❌ Indtast gyldig NACA 4-digit kode (root)")
        else:
            if root_upload is not None:
                try:
                    content = root_upload.getvalue().decode("utf-8", errors="ignore")
                    root_profile = parse_afl(content)
                except Exception as exc:
                    st.error(f"❌ Kunne ikke læse root-fil: {exc}")
            elif st.session_state.airfoil_profile is not None:
                root_profile = np.array(st.session_state.airfoil_profile)

        if tip_mode == "NACA 4-cifret":
            if is_valid_naca(tip_naca):
                spacing = st.session_state.geom_spacing
                closed_te = st.session_state.geom_closed_te
                tip_profile = naca_4_digit(tip_naca, num_points=120, spacing=spacing, closed_te=closed_te)
            else:
                st.error("❌ Indtast gyldig NACA 4-digit kode (tip)")
        else:
            if tip_upload is not None:
                try:
                    content = tip_upload.getvalue().decode("utf-8", errors="ignore")
                    tip_profile = parse_afl(content)
                except Exception as exc:
                    st.error(f"❌ Kunne ikke læse tip-fil: {exc}")
            elif root_profile is not None:
                tip_profile = np.array(root_profile)

        if root_profile is not None and tip_profile is not None:
            profile_name = f"{root_naca}→{tip_naca}" if not same_as_root else f"{root_naca}"
            blade_data = generate_blade_geometry(
                root_profile_2d=root_profile,
                tip_profile_2d=tip_profile,
                num_sections=num_sections,
                root_radius=root_radius,
                tip_radius=tip_radius,
                root_chord=root_chord,
                tip_chord=tip_chord,
                root_twist=root_twist,
                tip_twist=tip_twist,
                profile_name=profile_name,
                auto_spacing=auto_spacing
            )
            st.session_state.blade_data = blade_data
            st.success("✅ Blade geometri genereret!")

    st.subheader("⚡ Quick Optimize (Schmitz)")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        opt_tsr = st.number_input("Target TSR", 2.0, 12.0, 7.0, 0.5, key="opt_tsr")
    with col_opt2:
        alpha_opt = st.number_input("α opt [°]", 2.0, 12.0, 6.0, 0.5, key="opt_alpha")
    with col_opt3:
        cl_opt = st.number_input("Cl design", 0.6, 1.5, 1.0, 0.05, key="opt_cl")
    cd_opt = st.number_input("Cd design", 0.005, 0.05, 0.01, 0.005, key="opt_cd")
    blades = st.slider("Antal blades", 2, 4, 3, 1, key="opt_blades")

    if st.button("Quick Optimize", key="quick_optimize"):
        if root_profile is None or tip_profile is None:
            st.error("❌ Vælg root/tip airfoil først")
        else:
            blade_data = schmitz_optimize_geometry(
                root_profile=root_profile,
                tip_profile=tip_profile,
                hub_radius=root_radius,
                tip_radius=tip_radius,
                num_sections=num_sections,
                target_tsr=opt_tsr,
                alpha_opt_deg=alpha_opt,
                cl_design=cl_opt,
                cd_design=cd_opt,
                blades=blades,
                root_chord=root_chord,
                tip_chord=tip_chord,
                root_twist=root_twist,
                tip_twist=tip_twist
            )
            st.session_state.blade_data = blade_data
            st.success("✅ Quick Optimize færdig")
    
    if st.session_state.blade_data:
        blade = st.session_state.blade_data
        
        # Visualisering
        st.subheader("📊 Blade Geometri Oversigt")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Radiale Sektioner", blade['num_sections'])
        col2.metric("Root Radius", f"{blade['radii'][0]:.2f} m")
        col3.metric("Tip Radius", f"{blade['radii'][-1]:.2f} m")
        col4.metric("NACA Profil", blade['naca_code'])
        
        # Chord og Twist distributioner
        col1, col2 = st.columns(2)
        
        with col1:
            fig_chord = go.Figure()
            fig_chord.add_trace(go.Scatter(
                x=blade['radii'],
                y=blade['chords'],
                mode='lines+markers',
                name='Chord',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                hovertemplate='Radius: %{x:.3f} m<br>Chord: %{y:.3f} m<extra></extra>'
            ))
            fig_chord.update_layout(
                title='Chord Distribution',
                xaxis_title='Radius [m]',
                yaxis_title='Chord [m]',
                height=400
            )
            st.plotly_chart(fig_chord, use_container_width=True)
        
        with col2:
            fig_twist = go.Figure()
            fig_twist.add_trace(go.Scatter(
                x=blade['radii'],
                y=blade['twists'],
                mode='lines+markers',
                name='Twist',
                line=dict(color='red', width=3),
                marker=dict(size=8),
                hovertemplate='Radius: %{x:.3f} m<br>Twist: %{y:.2f}°<extra></extra>'
            ))
            fig_twist.update_layout(
                title='Twist Distribution',
                xaxis_title='Radius [m]',
                yaxis_title='Twist [°]',
                height=400
            )
            st.plotly_chart(fig_twist, use_container_width=True)

        st.subheader("🧊 3D Blade Preview")
        render_blade_3d(blade, title="Blade 3D Preview")
        
        # Export muligheder
        st.subheader("💾 Eksporter til YBlade_v2")
        
        col1, col2 = st.columns(2)
        
        with col1:
            yblade_csv = export_yblade_csv(blade)
            st.download_button(
                label="📥 Download YBlade CSV",
                data=yblade_csv,
                file_name=f"blade_{blade['naca_code']}.csv",
                mime="text/csv"
            )
        
        with col2:
            xyz_csv = export_xyz_simple(blade)
            st.download_button(
                label="📥 Download XYZ Koordinater",
                data=xyz_csv,
                file_name=f"blade_xyz_{blade['naca_code']}.csv",
                mime="text/csv"
            )

        fusion_csv = export_fusion_spline_csv(blade)
        st.download_button(
            label="📥 Download Fusion 360 Spline CSV",
            data=fusion_csv,
            file_name=f"blade_fusion_{blade['naca_code']}.csv",
            mime="text/csv"
        )

# ============================================================================
# FASE 3: OPTIMERING
# ============================================================================

elif phase == "Fase 3: Optimering":
    st.header("Fase 3: Blade Optimering for Betz Grænse")
    
    if 'blade_data' not in st.session_state or st.session_state.blade_data is None:
        st.warning("⚠️ Generer først blade geometri i Fase 2!")
        st.info(f"Seneste profil: {st.session_state.airfoil_name or 'Ingen'}")
    else:
        blade = st.session_state.blade_data
        st.caption(f"Aktiv profil: {blade.get('naca_code', 'Ukendt')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wind_speed = st.number_input("Vindhastighed [m/s]:", 5, 20, 10, 1)
        with col2:
            target_tsr = st.number_input("Target TSR:", 1.0, 10.0, 7.0, 0.5)
        with col3:
            rpm_range = st.slider("RPM Range:", 50, 500, (100, 300), 10)

        tol = st.number_input("Konvergenstolerance", 1e-7, 1e-3, 1e-5, format="%.1e")
        
        if st.button("⚡ Optimer Blade", key="optimize"):
            with st.spinner("Optimerer..."):
                # BEM analyse for forskellige RPM
                results_list = []
                for rpm in range(rpm_range[0], rpm_range[1], 10):
                    bem_result = bem_analysis(blade, wind_speed=wind_speed, rpm=rpm, tol=tol)
                    results_list.append({
                        'rpm': rpm,
                        'tsr': bem_result['tsr'],
                        'cp': bem_result['cp'],
                        'power': bem_result['total_power']
                    })
                
                df_opt = pd.DataFrame(results_list)
                
                # Find optimal
                best_idx = df_opt['cp'].idxmax()
                best_row = df_opt.loc[best_idx]
                
                st.subheader("⭐ Optimeringsresultater")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Optimal RPM", int(best_row['rpm']))
                col2.metric("TSR", f"{best_row['tsr']:.2f}")
                col3.metric("Cp (% af Betz)", f"{min(best_row['cp']/0.593, 1.0)*100:.1f}%")
                col4.metric("Effekt", f"{best_row['power']:.0f} W")
                
                # Plot Cp vs RPM
                fig_opt = go.Figure()
                fig_opt.add_trace(go.Scatter(
                    x=df_opt['rpm'],
                    y=df_opt['cp'],
                    mode='lines+markers',
                    name='Cp',
                    line=dict(color='green', width=3),
                    marker=dict(size=8),
                    hovertemplate='RPM: %{x}<br>Cp: %{y:.4f}<extra></extra>'
                ))
                fig_opt.add_hline(y=0.593, line_dash="dash", line_color="red", 
                                 annotation_text="Betz grænse (59.3%)")
                fig_opt.update_layout(
                    title='Cp vs RPM Optimering',
                    xaxis_title='RPM',
                    yaxis_title='Cp (Power Coefficient)',
                    height=500
                )
                st.plotly_chart(fig_opt, use_container_width=True)

                st.subheader("🧊 3D Blade Preview")
                render_blade_3d(blade, title="Blade 3D Preview")
                
                # Detaljeret BEM for optimal RPM
                st.subheader("📈 BEM Analyse ved Optimal RPM")
                best_bem = bem_analysis(blade, wind_speed=wind_speed, rpm=int(best_row['rpm']))
                
                bem_df = pd.DataFrame(best_bem['sections'])
                bem_df = bem_df[['radius', 'twist', 'chord', 'v_rel', 'alpha', 'cl', 'cd', 'power_section']]
                st.dataframe(bem_df, use_container_width=True)
                
                # Downloadable report
                report = f"""
BLADE OPTIMERINGSRAPPORT
========================
NACA Profil: {blade['naca_code']}
Antal Sektioner: {blade['num_sections']}
Vindhastighed: {wind_speed} m/s

OPTIMALE PARAMETRE:
- RPM: {int(best_row['rpm'])}
- TSR: {best_row['tsr']:.2f}
- Cp: {best_row['cp']:.4f} ({min(best_row['cp']/0.593, 1.0)*100:.1f}% af Betz grænse)
- Effekt: {best_row['power']:.1f} W

ROOT GEOMETRI:
- Radius: {blade['radii'][0]:.3f} m
- Chord: {blade['chords'][0]:.3f} m
- Twist: {blade['twists'][0]:.1f}°

TIP GEOMETRI:
- Radius: {blade['radii'][-1]:.3f} m
- Chord: {blade['chords'][-1]:.3f} m
- Twist: {blade['twists'][-1]:.1f}°
"""
                
                st.download_button(
                    label="📄 Download Optimeringssrapport",
                    data=report,
                    file_name="blade_optimization_report.txt"
                )

st.sidebar.markdown("---")
st.sidebar.markdown("### Om Appen")
st.sidebar.info(
    "**Fase 1-3 Blade Designer**\n\n"
    "- Fase 1: NACA profil generator\n"
    "- Fase 2: Blade geometri editor\n"
    "- Fase 3: BEM optimering\n\n"
    "Eksporter til YBlade_v2 for Fusion 360!"
)
