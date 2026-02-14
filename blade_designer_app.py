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

def naca_4_digit(code, num_points=100):
    """Genererer NACA 4-digit airfoil koordinater"""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:4]) / 100.0
    
    x = np.linspace(0, 1, num_points)
    
    # Tykkelsesfordeling (thickness distribution)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                  0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber linje
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
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


def generate_blade_geometry(naca_code, num_sections=10, root_radius=0.1, 
                           tip_radius=2.0, root_chord=0.3, tip_chord=0.1, 
                           root_twist=25, tip_twist=0):
    """Genererer 3D blade geometri med twist og taper"""
    
    radii = np.linspace(root_radius, tip_radius, num_sections)
    chords = np.linspace(root_chord, tip_chord, num_sections)
    twists = np.linspace(root_twist, tip_twist, num_sections)
    
    profile_2d = naca_4_digit(naca_code, num_points=50)
    
    blade_sections = []
    for i, (r, chord, twist) in enumerate(zip(radii, chords, twists)):
        scaled_profile = profile_2d * chord
        
        twist_rad = np.radians(twist)
        rotated_profile = scaled_profile.copy()
        rotated_profile[:, 1] = scaled_profile[:, 1] * np.cos(twist_rad) - r * np.sin(twist_rad)
        
        blade_sections.append({
            'radius': r,
            'chord': chord,
            'twist': twist,
            'profile': rotated_profile
        })
    
    return {
        'naca_code': naca_code,
        'num_sections': num_sections,
        'radii': radii,
        'chords': chords,
        'twists': twists,
        'sections': blade_sections,
        'profile_2d': profile_2d
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


# ============================================================================
# FASE 2: SIMPLE BEM BEREGNINGER
# ============================================================================

def bem_analysis(blade_data, wind_speed=10, rpm=100, rho=1.225, tsr=None):
    """
    Blade Element Momentum teori - simpel version
    Beregner kraft og effekt på hver blade sektion
    """
    
    if tsr is None:
        avg_radius = np.mean(blade_data['radii'])
        omega = rpm * 2 * np.pi / 60
        tsr = (omega * avg_radius) / wind_speed
    
    results = []
    
    for section in blade_data['sections']:
        r = section['radius']
        chord = section['chord']
        twist = section['twist']
        
        # Induction factors (simpelt)
        a = 0.33  # Axial induction
        a_prime = 0.05  # Angular induction
        
        omega = rpm * 2 * np.pi / 60
        v_rel = np.sqrt((wind_speed * (1 - a))**2 + (omega * r * (1 + a_prime))**2)
        
        # Angle of attack
        phi = np.arctan2(wind_speed * (1 - a), omega * r * (1 + a_prime))
        alpha = np.degrees(phi) - twist
        
        # Simpel Cl/Cd (approximation)
        alpha_rad = np.radians(alpha)
        cl = 1.5 * np.sin(2 * alpha_rad)
        cd = 0.01 + 0.02 * np.sin(alpha_rad)**2
        
        # Kræfter
        lift = 0.5 * rho * v_rel**2 * chord * cl
        drag = 0.5 * rho * v_rel**2 * chord * cd
        
        # Tangential og normal komponenter
        fn = lift * np.cos(phi) + drag * np.sin(phi)
        ft = lift * np.sin(phi) - drag * np.cos(phi)
        
        # Torque på sektion
        dr = blade_data['radii'][1] - blade_data['radii'][0] if len(blade_data['radii']) > 1 else 0.1
        torque = ft * r * dr * 3  # 3 blades
        power = torque * omega
        
        results.append({
            'radius': r,
            'twist': twist,
            'chord': chord,
            'v_rel': v_rel,
            'alpha': alpha,
            'cl': cl,
            'cd': cd,
            'lift': lift,
            'drag': drag,
            'fn': fn,
            'ft': ft,
            'torque_section': torque,
            'power_section': power
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


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Blade Designer", layout="wide")
st.title("🌪️ Vindmølle Blade Designer - Fase 1-3")

# Sidebar navigation
st.sidebar.markdown("### Navigation")
phase = st.sidebar.radio("Vælg fase:", ["Fase 1: NACA Generator", "Fase 2: Geometri Editor", "Fase 3: Optimering"])

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
        
        naca_input = st.text_input("Indtast NACA kode:", "2412", max_chars=4)
        
        # Valider input
        if len(naca_input) == 4 and naca_input.isdigit():
            num_points = st.slider("Antal punkter langs profil:", 50, 200, 100)
            
            # Generér profil
            profile = naca_4_digit(naca_input, num_points)
            
            # Plot med Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=profile[:, 0],
                y=profile[:, 1],
                fill='toself',
                name='NACA Profil',
                mode='lines',
                line=dict(color='blue', width=2),
                hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'NACA {naca_input} Airfoil Profil',
                xaxis_title='Chord Position (x/c)',
                yaxis_title='Tykkelse (y/c)',
                height=500,
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Profil statistik
            col1a, col1b, col1c = st.columns(3)
            col1a.metric("Max Camber", f"{int(naca_input[0])}%")
            col1b.metric("Camber Position", f"{int(naca_input[1])} x 10%")
            col1c.metric("Tykkelse", f"{naca_input[2:4]}%")
        else:
            st.error("❌ Indtast gyldig NACA 4-digit kode (f.eks. 2412)")

# ============================================================================
# FASE 2: GEOMETRI EDITOR
# ============================================================================

elif phase == "Fase 2: Geometri Editor":
    st.header("Fase 2: Blade Geometri Designer")
    
    # Initialize session state
    if 'blade_data' not in st.session_state:
        st.session_state.blade_data = None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔧 Profil Indstillinger")
        naca = st.text_input("NACA Kode:", "2412", max_chars=4)
        num_sections = st.slider("Antal radiale sektioner:", 5, 25, 12)
    
    with col2:
        st.subheader("📏 Blade Dimensioner")
        root_radius = st.number_input("Rod radius [m]:", 0.05, 1.0, 0.15, 0.05)
        tip_radius = st.number_input("Tip radius [m]:", 0.5, 5.0, 2.0, 0.1)
    
    with col3:
        st.subheader("✈️ Chord & Twist")
        root_chord = st.number_input("Rod chord [m]:", 0.1, 1.0, 0.4, 0.05)
        tip_chord = st.number_input("Tip chord [m]:", 0.01, 0.5, 0.05, 0.01)
        root_twist = st.number_input("Rod twist [°]:", 0, 45, 25, 1)
        tip_twist = st.number_input("Tip twist [°]:", -10, 10, 0, 1)
    
    if st.button("🔨 Generer Blade Geometri", key="gen_blade"):
        blade_data = generate_blade_geometry(
            naca_code=naca,
            num_sections=num_sections,
            root_radius=root_radius,
            tip_radius=tip_radius,
            root_chord=root_chord,
            tip_chord=tip_chord,
            root_twist=root_twist,
            tip_twist=tip_twist
        )
        st.session_state.blade_data = blade_data
        st.success("✅ Blade geometri genereret!")
    
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
        
        # Export muligheder
        st.subheader("💾 Eksporter til YBlade_v2")
        
        col1, col2 = st.columns(2)
        
        with col1:
            yblade_csv = export_yblade_csv(blade)
            st.download_button(
                label="📥 Download YBlade CSV",
                data=yblade_csv,
                file_name=f"blade_{naca}.csv",
                mime="text/csv"
            )
        
        with col2:
            xyz_csv = export_xyz_simple(blade)
            st.download_button(
                label="📥 Download XYZ Koordinater",
                data=xyz_csv,
                file_name=f"blade_xyz_{naca}.csv",
                mime="text/csv"
            )

# ============================================================================
# FASE 3: OPTIMERING
# ============================================================================

elif phase == "Fase 3: Optimering":
    st.header("Fase 3: Blade Optimering for Betz Grænse")
    
    if 'blade_data' not in st.session_state or st.session_state.blade_data is None:
        st.warning("⚠️ Generer først blade geometri i Fase 2!")
    else:
        blade = st.session_state.blade_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wind_speed = st.number_input("Vindhastighed [m/s]:", 5, 20, 10, 1)
        with col2:
            target_tsr = st.number_input("Target TSR:", 1.0, 10.0, 7.0, 0.5)
        with col3:
            rpm_range = st.slider("RPM Range:", 50, 500, (100, 300), 10)
        
        if st.button("⚡ Optimer Blade", key="optimize"):
            with st.spinner("Optimerer..."):
                # BEM analyse for forskellige RPM
                results_list = []
                for rpm in range(rpm_range[0], rpm_range[1], 10):
                    bem_result = bem_analysis(blade, wind_speed=wind_speed, rpm=rpm)
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
