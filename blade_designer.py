"""
Fase 1: NACA Blade Designer for YBlade_v2 Integration
Genererer NACA airfoil profiler og blade geometri for Fusion 360 export
"""

import numpy as np
import pandas as pd
from io import StringIO


def naca_4_digit(code, num_points=100):
    """
    Genererer NACA 4-digit airfoil koordinater
    Ex: "2412" = 2% camber, 4/10 position, 12% tykkelse
    
    Args:
        code: String "MPTC" hvor M=max camber%, P=position/10, T=tykkelse%
        num_points: Antal punkter langs chord
    
    Returns:
        array med [x, y] koordinater
    """
    m = int(code[0]) / 100.0  # Max camber
    p = int(code[1]) / 10.0   # Camber position
    t = int(code[2:4]) / 100.0  # Tykkelse
    
    x = np.linspace(0, 1, num_points)
    
    # Tykkelsesfordeling
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
    
    # Øvre og nedre overflade
    xu = x - yt * np.sin(theta)
    xl = x + yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    yl = yc - yt * np.cos(theta)
    
    # Kombiner øvre og nedre side
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    
    return np.column_stack([x_coords, y_coords])


def generate_blade_geometry(naca_code, num_sections=10, root_radius=0.1, 
                           tip_radius=2.0, root_chord=0.3, tip_chord=0.1, 
                           root_twist=25, tip_twist=0):
    """
    Genererer 3D blade geometri med variabel twist og chord
    
    Args:
        naca_code: NACA profil (f.eks. "2412")
        num_sections: Antal radiale sektioner
        root_radius: Blade rod radius [m]
        tip_radius: Blade tip radius [m]
        root_chord: Blade rod chord [m]
        tip_chord: Blade tip chord [m]
        root_twist: Twist ved rod [grader]
        tip_twist: Twist ved tip [grader]
    
    Returns:
        Dict med blade data
    """
    # Radiale positoner
    radii = np.linspace(root_radius, tip_radius, num_sections)
    
    # Interpoler chord og twist
    chords = np.linspace(root_chord, tip_chord, num_sections)
    twists = np.linspace(root_twist, tip_twist, num_sections)
    
    # Generer profil
    profile_2d = naca_4_digit(naca_code, num_points=50)
    
    blade_sections = []
    for i, (r, chord, twist) in enumerate(zip(radii, chords, twists)):
        # Skalér profil til aktuelt chord
        scaled_profile = profile_2d * chord
        
        # Appliker twist (rotation omkring x-axis)
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


def export_yblade_csv(blade_data, filename="blade_coordinates.csv"):
    """
    Eksporterer blade koordinater i YBlade_v2 kompatibelt CSV format
    
    Forventet format for YBlade_v2:
    radius,chord,twist,x1,y1,x2,y2,...,xN,yN
    """
    rows = []
    
    for section in blade_data['sections']:
        profile = section['profile']
        
        # Byg CSV-række
        row = {
            'radius_m': section['radius'],
            'chord_m': section['chord'],
            'twist_deg': section['twist']
        }
        
        # Tilføj alle profil-koordinater
        for j, (x, y) in enumerate(profile):
            row[f'x{j}'] = x
            row[f'y{j}'] = y
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_str = df.to_csv(index=False)
    
    return csv_str, df


def export_simple_xyz(blade_data):
    """
    Simpel XYZ export for manuel Fusion 360 import
    Format: radius, x_chord, y_thickness, z_twist
    """
    data = []
    
    for section in blade_data['sections']:
        profile = section['profile']
        
        for x, y in profile:
            data.append({
                'radius': section['radius'],
                'x_chord': x,
                'y_thickness': y,
                'z_twist': section['twist']
            })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False), df


def calculate_blade_properties(blade_data, rho=1.225, rpm=100):
    """
    Beregner blade properties for CFD/BEM analyse
    """
    total_volume = 0
    total_area = 0
    
    for section in blade_data['sections']:
        profile = section['profile']
        chord = section['chord']
        
        # Område af profil (simpel trapezoidal integration)
        area_2d = np.trapz(np.abs(profile[:, 1]), profile[:, 0])
        total_area += area_2d * chord
    
    # Gennemsnitlige egenskaber
    avg_radius = np.mean(blade_data['radii'])
    avg_chord = np.mean(blade_data['chords'])
    avg_twist = np.mean(blade_data['twists'])
    
    # Omtrentlig blade masse (assuming aluminum, ~2700 kg/m³)
    blade_mass = total_area * 0.01 * 2700  # 10mm tykkelse antaget
    
    return {
        'total_blade_area': total_area,
        'avg_radius': avg_radius,
        'avg_chord': avg_chord,
        'avg_twist': avg_twist,
        'estimated_mass': blade_mass,
        'rpm': rpm,
        'tip_speed': avg_radius * (rpm * 2 * np.pi / 60)
    }


if __name__ == "__main__":
    # Test
    blade = generate_blade_geometry(
        naca_code="2412",
        num_sections=15,
        root_radius=0.15,
        tip_radius=2.0,
        root_chord=0.4,
        tip_chord=0.05
    )
    
    print("Blade genereret!")
    print(f"Profiler: {len(blade['sections'])}")
    print(f"Root: {blade['radii'][0]:.2f}m, Chord: {blade['chords'][0]:.2f}m")
    print(f"Tip: {blade['radii'][-1]:.2f}m, Chord: {blade['chords'][-1]:.2f}m")
    
    # Test export
    csv_str, df = export_yblade_csv(blade)
    print("\nYBlade CSV Preview:")
    print(df.head())
    
    # Test properties
    props = calculate_blade_properties(blade)
    print("\nBlade Properties:")
    for key, val in props.items():
        print(f"  {key}: {val:.2f}")
