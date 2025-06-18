# ===== utils/physics.py =====
import numpy as np

def calculate_stress_intensity(cutting_params):
    """
    Calculate stress intensity factor from cutting parameters.
    
    Args:
        cutting_params: Dict containing cutting parameters
    
    Returns:
        Stress intensity factor (MPa√m)
    """
    # Extract parameters
    cutting_speed = cutting_params.get('cutting_speed', 100)  # m/min
    feed_rate = cutting_params.get('feed_rate', 0.1)  # mm/rev
    depth_of_cut = cutting_params.get('depth_of_cut', 1.0)  # mm
    tool_wear = cutting_params.get('tool_wear', 0.1)  # mm
    
    # Simplified stress intensity calculation
    # In practice, this would be based on more complex mechanics
    stress = 100 * np.sqrt(cutting_speed * feed_rate)  # MPa
    crack_length = depth_of_cut * tool_wear * 0.1  # mm
    
    # K = σ * √(π * a)
    K = stress * np.sqrt(np.pi * crack_length / 1000)  # Convert mm to m
    
    return K

def paris_law_prediction(delta_K, C=1e-10, m=3.0, N_cycles=1000):
    """
    Predict crack growth using Paris law.
    
    Args:
        delta_K: Stress intensity factor range (MPa√m)
        C: Paris law coefficient
        m: Paris law exponent
        N_cycles: Number of cycles
    
    Returns:
        Predicted crack growth (mm)
    """
    # da/dN = C * (ΔK)^m
    growth_rate = C * (delta_K ** m)  # m/cycle
    
    # Total growth
    total_growth = growth_rate * N_cycles * 1000  # Convert to mm
    
    return total_growth

def calculate_material_factor(material_type):
    """Get material-specific factor for crack prediction."""
    material_factors = {
        'steel': 1.0,
        'aluminum': 0.7,
        'titanium': 1.3,
        'inconel': 1.5,
        'cast_iron': 1.1,
        'stainless_steel': 1.2
    }
    return material_factors.get(material_type.lower(), 1.0)

def calculate_tool_factor(tool_type):
    """Get tool-specific factor for crack prediction."""
    tool_factors = {
        'carbide': 0.8,
        'hss': 1.2,
        'ceramic': 0.9,
        'cbn': 0.7,
        'diamond': 0.6,
        'coated_carbide': 0.75
    }
    return tool_factors.get(tool_type.lower(), 1.0)