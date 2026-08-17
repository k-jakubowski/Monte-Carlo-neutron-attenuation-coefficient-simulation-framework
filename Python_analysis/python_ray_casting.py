import numpy as np
import uproot
import trimesh
import trimesh.transformations as tf
import math
import multiprocessing as mp
from functools import partial
import os
import pyvista
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import brentq

# %%
# =============================================================================
# === USER INPUTS & CONFIGURATION ===
# =============================================================================

SAMPLE_CHOICE = "wire"  # Choose from 'solder', 'wire', or 'gilding'

# --- ROI Configuration ---
RAY_RESOLUTION = 512

ROTATE_MESH = True 
ROTATION_AXIS = [0,0,1]  

# File Paths
if SAMPLE_CHOICE == "solder":
    INPUT_STL = "/path_to/solder.stl"
    OPEN_BEAM_FILE = "/path_to/open_beam_solder_run1.root"
    RESULTS_FILE = "/path_to/solder_run1.root"

    OUTPUT_PREFIX = "/path_to/solder_data"

    # Bottom of the sample - 1.5 mm x 1.5 mm ROI
    ROI_SIZE_X_MM = 3.0 
    ROI_SIZE_Y_MM = 1.5  
    ROI_SIZE_Z_MM = 1.5  

    ROI_CENTER_X_MM = 0.0
    ROI_CENTER_Y_MM = 0.0
    ROI_CENTER_Z_MM = -0.5
    
    # Bottom of the sample - 0.3 mm x 0.3 mm ROI
    '''
    ROI_SIZE_X_MM = 3.0
    ROI_SIZE_Y_MM = 0.3  
    ROI_SIZE_Z_MM = 0.3  

    ROI_CENTER_X_MM = 0.0
    ROI_CENTER_Y_MM = 0.0
    ROI_CENTER_Z_MM = -0.2
    
    # Top of the sample - 1.5 mm x 1.5 mm ROI
    ROI_SIZE_X_MM = 3.0
    ROI_SIZE_Y_MM = 1.5  
    ROI_SIZE_Z_MM = 1.5  

    ROI_CENTER_X_MM = 0.0
    ROI_CENTER_Y_MM = 0.0
    ROI_CENTER_Z_MM = 1.0
    '''

    # Needed to align the sample with the ray direction in the analysis (matches the beam path in the simulation)
    ROTATION_ANGLE_DEG = 270.0
# %%

elif SAMPLE_CHOICE == "wire":
    INPUT_STL = "/path_to/wire.stl"
    OPEN_BEAM_FILE = "/path_to/open_beam_wire_run1.root"

    #RESULTS_FILE = "/Workspace/Dingo_medieval_cross/detector_build/Results/wire/wire_sample_1cm2_scoring_plane_1E6_run10.root"
    RESULTS_FILE = "path_to/wire_run1.root"
    OUTPUT_PREFIX = "/path_to/wire_data"

    # centre of the wire, 0.1 mm x 0.1 mm ROI
    ROI_SIZE_X_MM = 0.8
    ROI_SIZE_Y_MM = 0.1
    ROI_SIZE_Z_MM = 0.1

    ROI_CENTER_X_MM = 0.0
    ROI_CENTER_Y_MM = 0.0
    ROI_CENTER_Z_MM = -0.04

    ROTATION_ANGLE_DEG = 180.0
# %%

elif SAMPLE_CHOICE=="gilding":
    INPUT_STL = "/path_to/gilding.stl"
    OPEN_BEAM_FILE = "/path_to/open_beam_gilding_run1.root"

    RESULTS_FILE = "/path_to/gilding_run1.root"
    OUTPUT_PREFIX = "/path_to/gilding_data"

    # 0.1 mm x 0.1 mm ROI
    ROI_SIZE_X_MM = 2.9
    ROI_SIZE_Y_MM = 0.1
    ROI_SIZE_Z_MM = 0.1

    # Top of the sample
    #ROI_CENTER_X_MM = 0.0
    #ROI_CENTER_Y_MM = 0.4
    #ROI_CENTER_Z_MM = 2.1

    # Central area of the sample
    ROI_CENTER_X_MM = 0.0
    ROI_CENTER_Y_MM = 0.3
    ROI_CENTER_Z_MM = 0.5

    ROTATION_ANGLE_DEG = 180.0

# Defined Macroscopic Coefficient (Sigma) ranges based on experimental mu values (cm^-1)
MATERIAL_RANGES = {
    'solder': (1.9, 2.1),
    'wire': (2.3, 2.5),
    'gilding': (3.0, 4.0)
}

# =============================================================================
# === THICKNESS MAP (Ray Casting) ===
# =============================================================================
# %%

def _calculate_object_thickness_for_chunk(ray_chunk_data, mesh, start_ray_axis):
    ray_origins, ray_directions, global_start_index = ray_chunk_data
    num_rays = ray_origins.shape[0]
    
    material_thicknesses = np.full(num_rays, np.nan)
    surface_locations = np.full((num_rays, 3), np.nan)

    # Perform intersection query against the mesh
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins, ray_directions, multiple_hits=True
    )
    
    unique_rays, counts = np.unique(index_ray, return_counts=True)
    for ray_idx, hit_count in zip(unique_rays, counts):
        # Only process rays that enter and exit the mesh (even number of hits >= 2) - excludes voids
        if hit_count >= 2 and hit_count % 2 == 0:
            ray_hits_mask = index_ray == ray_idx
            ray_hits = locations[ray_hits_mask]
            
            # Sort hits by distance from origin to pair entry/exit points correctly
            distances = np.linalg.norm(ray_hits - ray_origins[ray_idx], axis=1)
            sorted_indices = np.argsort(distances)
            sorted_hits = ray_hits[sorted_indices]
            
            total_material_thickness = 0.0
            # Accumulate thickness between successive entry and exit points
            for j in range(0, hit_count, 2):
                entry_point = sorted_hits[j]
                exit_point = sorted_hits[j+1]
                total_material_thickness += np.linalg.norm(exit_point - entry_point)
                
            material_thicknesses[ray_idx] = total_material_thickness
            
            # Store the midpoint of the first and last hit as the nominal surface location
            first_hit = sorted_hits[0]
            last_hit = sorted_hits[-1]
            surface_locations[ray_idx] = (first_hit + last_hit) / 2
            
    global_indices = global_start_index + np.arange(num_rays)
    return material_thicknesses, surface_locations, global_indices
# %%

def generate_thickness_maps(stl_path, resolution=512, n_cores=None):
    """
    Loads an STL, rotates it to align with the beam, and casts a grid of parallel rays to build a 2D thickness map of the object.
    
    Args:
        stl_path (str): Filepath to the STL mesh.
        resolution (int): The grid resolution (N x N) for ray casting.
        n_cores (int, optional): Number of CPU cores to use. Defaults to all available cores - 1.
        
    Returns:
        tuple: (mesh, full_locations_mm, full_thicknesses_mm, roi_thicknesses_mm, roi_bounds_mm)
    """
        
    print(f"\nLoading mesh from {stl_path}...")
    mesh = trimesh.load(stl_path)
    
    if ROTATE_MESH:
        angle_rad = np.radians(ROTATION_ANGLE_DEG)
        rot_matrix = tf.rotation_matrix(angle_rad, ROTATION_AXIS)
        mesh.apply_transform(rot_matrix)
        print(f"Rotated mesh by {ROTATION_ANGLE_DEG} degrees.")
        
    bounds = mesh.bounds
    min_b, max_b = bounds
    
    # Beam alignment: Casting along X-axis
    start_ray_axis = 0       
    ax1_idx, ax2_idx = 1, 2  
    
    coords_1 = np.linspace(min_b[ax1_idx], max_b[ax1_idx], resolution)
    coords_2 = np.linspace(min_b[ax2_idx], max_b[ax2_idx], resolution)
    grid_1, grid_2 = np.meshgrid(coords_1, coords_2)
    
    # Create a uniform grid of rays spanning the bounding box of the mesh
    ray_origins = np.zeros((grid_1.size, 3))
    ray_origins[:, ax1_idx] = grid_1.flatten() # Y positions
    ray_origins[:, ax2_idx] = grid_2.flatten() # Z positions
    ray_origins[:, start_ray_axis] = max_b[start_ray_axis] + 1.0

    # Set direction vector to [-1, 0, 0] to cast back towards the mesh
    ray_directions = np.zeros((grid_1.size, 3))
    ray_directions[:, start_ray_axis] = -1.0
    
    total_rays = ray_origins.shape[0]
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)
        
    chunk_size = math.ceil(total_rays / n_cores)
    chunk_data = []
    for i in range(0, total_rays, chunk_size):
        end_idx = min(i + chunk_size, total_rays)
        chunk_data.append((ray_origins[i:end_idx], ray_directions[i:end_idx], i))
        
    print(f"Casting {total_rays} rays along X-axis over the FULL object...")

    # Multiprocessing
    pool = mp.Pool(processes=n_cores)
    partial_chunk_func = partial(_calculate_object_thickness_for_chunk, mesh=mesh, start_ray_axis=start_ray_axis)
    results = pool.map(partial_chunk_func, chunk_data)
    pool.close()
    pool.join()
    
    # Reassemble chunked results
    final_thicknesses = np.full(total_rays, np.nan)
    final_locations = np.full((total_rays, 3), np.nan)
    
    for thicknesses, locations, global_indices in results:
        valid_mask = ~np.isnan(thicknesses)
        if np.any(valid_mask):
            final_thicknesses[global_indices[valid_mask]] = thicknesses[valid_mask]
            final_locations[global_indices[valid_mask]] = locations[valid_mask]
            
    valid_mask = ~np.isnan(final_thicknesses)
    full_locations_mm = final_locations[valid_mask] * 1000.0
    full_thicknesses_mm = final_thicknesses[valid_mask] * 1000.0

    # Filter data to extract only the Region of Interest (ROI)
    roi_y_min_mm = ROI_CENTER_Y_MM - (ROI_SIZE_Y_MM / 2.0)
    roi_y_max_mm = ROI_CENTER_Y_MM + (ROI_SIZE_Y_MM / 2.0)
    roi_z_min_mm = ROI_CENTER_Z_MM - (ROI_SIZE_Z_MM / 2.0)
    roi_z_max_mm = ROI_CENTER_Z_MM + (ROI_SIZE_Z_MM / 2.0)
    
    roi_mask = (
        (full_locations_mm[:, 1] >= roi_y_min_mm) & (full_locations_mm[:, 1] <= roi_y_max_mm) &
        (full_locations_mm[:, 2] >= roi_z_min_mm) & (full_locations_mm[:, 2] <= roi_z_max_mm)
    )
    
    roi_thicknesses_mm = full_thicknesses_mm[roi_mask]
    
    # ROI bounds for the red box visualization
    roi_x_min_mm = ROI_CENTER_X_MM - (ROI_SIZE_X_MM / 2.0)
    roi_x_max_mm = ROI_CENTER_X_MM + (ROI_SIZE_X_MM / 2.0)
    roi_bounds_mm = (roi_x_min_mm, roi_x_max_mm, roi_y_min_mm, roi_y_max_mm, roi_z_min_mm, roi_z_max_mm)

    return mesh, full_locations_mm, full_thicknesses_mm, roi_thicknesses_mm, roi_bounds_mm

# =============================================================================
# === RAY-BY-RAY ROOT & SIMULATION ANALYSIS ===
# =============================================================================
# %%

def ray_by_ray_transmission(sigma, thicknesses_cm):
    """
    Calculates total theoretical transmission given a specific macroscopic cross-section (Sigma) and an array of ray thicknesses using the Beer-Lambert law.
    
    Args:
        sigma (float): Macroscopic cross-section in cm^-1.
        thicknesses_cm (np.ndarray): Array of ray thicknesses in cm.
        
    Returns:
        float: Expected total transmission fraction.
    """

    return np.mean(np.exp(-sigma * thicknesses_cm))
# %%

def calculate_simulation_results(open_file_path, results_file_path, roi_thicknesses_mm, sample_choice, target_sigma_range):
    """
    Loads simulation results, calculates overall transmission, and uses a root-finding algorithm to deduce the effective macroscopic cross-section.
    
    Args:
        open_file_path (str): Path to the open beam (unattenuated) ROOT file.
        results_file_path (str): Path to the sample (attenuated) ROOT file.
        roi_thicknesses_mm (np.ndarray): Material thicknesses from the ray casting step.
        sample_choice (str): The name of the sample.
        target_sigma_range (tuple): Expected empirical bounds for Sigma.
        
    Returns:
        float: The converged simulated macroscopic cross section (Sigma).
    """

    print(f"\n--- Processing ROOT Simulation Data ---")
    
    try:
        open_file = uproot.open(open_file_path)
        results_file = uproot.open(results_file_path)
        # Assuming single-branch tree arrays containing kinetic energy
        energies_open = open_file["NeutronEnergyData/NeutronKE_MeV"].array()
        energies_results = results_file["NeutronEnergyData/NeutronKE_MeV"].array()
    except Exception as e:
        print(f"ERROR: Could not load ROOT file arrays. {e}")
        return None
        
    counts_open = len(energies_open)
    counts_sample = len(energies_results)
    
    if counts_open == 0: 
        print("ERROR: Open beam counts are zero.")
        return None
        
    # Simulated transmission
    transmission_simulated = counts_sample / counts_open
    
    # Convert ROI thickness map to cm for cross-section calculations
    roi_thicknesses_cm = roi_thicknesses_mm / 10.0
    
    # ROOT FINDING (Bisection/Brent's Method)
    sigma_simulated = np.nan
    if 0 < transmission_simulated < 1:
        # Define the objective function for the root solver
        # Theoretical Transmission - Simulated Transmission = 0
        def objective(sigma):
            return ray_by_ray_transmission(sigma, roi_thicknesses_cm) - transmission_simulated
        
        try:
            # Solve for Sigma. Bounds (0.001 to 20.0 cm^-1) cover realistic thermal/cold neutron macroscopic cross sections.
            sigma_simulated = brentq(objective, 0.001, 20.0)
        except ValueError:
            print("ERROR: Could not converge on a Sigma value. Transmission might be out of physical bounds for this thickness.")
    
    # Calculate theoretical transmission bounds using the experimental Sigma range
    sigma_min, sigma_max = target_sigma_range
    # Using the ray-by-ray forward model to find expected transmission
    expected_t_max = ray_by_ray_transmission(sigma_min, roi_thicknesses_cm) # lower sigma = higher transmission
    expected_t_min = ray_by_ray_transmission(sigma_max, roi_thicknesses_cm) # higher sigma = lower transmission
    
    print(f"\n=======================================================")
    print(f"=== RAY-BY-RAY ANALYSIS RESULTS: {sample_choice.upper()} ROI ===")
    print(f"=======================================================")
    print(f"Target Experimental Σ Range : {sigma_min:.2f} to {sigma_max:.2f} cm^-1")
    print(f"Expected Transmission Range : {expected_t_min:.5f} to {expected_t_max:.5f} (Based on Geometry)")
    print(f"-------------------------------------------------------")
    print(f"Open Beam Counts            : {counts_open}")
    print(f"Sample Beam Counts          : {counts_sample}")
    print(f"-------------------------------------------------------")
    print(f"Simulated Transmission (T)  : {transmission_simulated:.5f}")
    print(f"Simulated Target Sigma (Σ)  : {sigma_simulated:.5f} cm^-1")
    print(f"=======================================================\n")
    
    return sigma_simulated

# =============================================================================
# === Plotting Functions (Matplotlib & PyVista) ===
# =============================================================================
# %%

def plot_roi_projections(surface_locations, thicknesses, full_bounds, roi_bounds, output_path_prefix, vmin, vmax):
    """
        Generates XZ and YZ 2D scatter plots of the sample thickness with ROI overlay.
    """

    if len(thicknesses) == 0: return

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')
    
    min_b, max_b = full_bounds
    roi_x_min, roi_x_max, roi_y_min, roi_y_max, roi_z_min, roi_z_max = roi_bounds

    projections = [
        (surface_locations[:, 0], surface_locations[:, 2], 'XZ Projection with ROI Box', 'X (mm)', 'Z (mm)', '_XZ_Projection_ROI.png', 0, 2, 
            ('z_lines', roi_z_min, roi_z_max)
        ),
        (surface_locations[:, 1], surface_locations[:, 2], 'YZ Projection with ROI Box', 'Y (mm)', 'Z (mm)', '_YZ_Projection_ROI.png', 1, 2, 
            (roi_y_min, roi_y_max, roi_z_min, roi_z_max)
        )
    ]

    print("\n--- Generating 2D Thickness Projection Plots with ROI Box (Matplotlib) ---")
    for x_data, y_data, title, x_label, y_label, suffix, x_lim_idx, y_lim_idx, roi_def in projections:
        output_image_path = f"{output_path_prefix}{suffix}"
        #output_svg_image_path = output_image_path.replace('.png', '.pdf')

        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(x_data, y_data, c=thicknesses, cmap=cmap, norm=norm, s=1, marker='.')
        plt.gca().set_aspect('equal', adjustable='box')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Material Thickness (mm)', size=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        for label in cbar.ax.get_yticklabels(): label.set_fontweight('bold')
            
        plt.xlabel(x_label, fontsize=14, labelpad=9.0, fontweight='bold')
        plt.ylabel(y_label, fontsize=14, labelpad=9.0, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.xlim(min_b[x_lim_idx], max_b[x_lim_idx])
        plt.ylim(min_b[y_lim_idx], max_b[y_lim_idx])
            
        # Draw ROI overlay based on projection plane
        if len(roi_def) == 4: 
            y_min, y_max, z_min, z_max = roi_def
            roi_rect = plt.Rectangle((y_min, z_min), y_max - y_min, z_max - z_min, fill=False, edgecolor='red', linewidth=3, linestyle='--')
            plt.gca().add_patch(roi_rect)
        elif len(roi_def) == 3 and roi_def[0] == 'z_lines': 
            _, z_min, z_max = roi_def
            plt.axhline(z_min, color='red', linestyle='--', linewidth=3)
            plt.axhline(z_max, color='red', linestyle='--', linewidth=3)
        
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.05)

        # SVG figure that is generated is of large size
        #plt.savefig(output_svg_image_path, dpi='figure', format='svg', bbox_inches='tight', pad_inches=0.05, facecolor='auto', edgecolor='auto', transparent=True, backend='svg')

        plt.close()
# %%

def plot_thickness_statistics(thicknesses, output_path_prefix, vmin, vmax, title_suffix):
    """
        Generates a histogram of thicknesses with overlaid statistical markers.
    """

    file_suffix = title_suffix.replace(' ', '_').replace('/', '') if title_suffix else "Full_Object"
    output_image_path = f"{output_path_prefix}_Thickness_Statistics_{file_suffix}.png" 
    output_svg_image_path = output_image_path.replace('.png', '.svg')

    if len(thicknesses) == 0: return

    mean_thickness = np.mean(thicknesses)
    median_thickness = np.median(thicknesses)
    std_thickness = np.std(thicknesses)
    min_val = np.min(thicknesses)
    max_val = np.max(thicknesses)

    q75, q25 = np.percentile(thicknesses, [75 ,25])
    iqr = q75 - q25

    print(f"\n--- Statistical Summary ({title_suffix}) ---")
    print(f"Total Valid Measurements: {len(thicknesses)}")
    print(f"Mean Thickness: {mean_thickness:.2f} mm")
    print(f"Median Thickness: {median_thickness:.2f} mm")
    print(f"Std Dev: {std_thickness:.2f} mm")
    print(f"Interquartile Range (IQR): {iqr:.2f} mm (Q1: {q25:.2f}, Q3: {q75:.2f})")
    print(f"Min/Max (Filtered): {min_val:.2f} / {max_val:.2f} mm")

    plt.figure(figsize=(10, 6))
    plt.hist(thicknesses, bins=50, color='deepskyblue', edgecolor='black', alpha=0.7, range=(vmin, vmax))
    
    plt.axvline(mean_thickness, color='orangered', linestyle='dashed', linewidth=3, label=f'Mean: {mean_thickness:.2f} mm')
    plt.axvline(median_thickness, color='black', linestyle='dashed', linewidth=3, label=f'Median: {median_thickness:.2f} mm')
    plt.axvspan(q25, q75, color='yellow', alpha=0.45, label=f'IQR: {iqr:.2f} mm')
    
    stats_text = f"N={len(thicknesses)}\nMean: {mean_thickness:.2f} mm\nMedian: {median_thickness:.2f} mm\nStd Dev: {std_thickness:.2f} mm\nIQR: {iqr:.2f} mm"

    #plt.text(0.95, 0.85, stats_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
    #         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8), fontsize=12, fontweight='bold')
    
    plt.xlabel('Thickness (mm)', fontsize=14, labelpad=9.0, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, labelpad=9.0, fontweight='bold')
    plt.xlim(vmin, vmax)
    
    leg = plt.legend(loc='upper left', fontsize=10)
    for line in leg.get_lines(): line.set_linewidth(2) 

    plt.grid(axis='y', alpha=0.5)
    #plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(output_svg_image_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close()
# %%

def render_3d_thickness_map(mesh, full_locations_mm, full_thicknesses_mm, roi_bounds, vmin, vmax, output_path_prefix, off_screen=True):
    """
        Renders a 3D visualization mapping thicknesses to mesh vertices using PyVista.
    """

    output_image_path = f"{output_path_prefix}_3D_Render_PyVista.png"

    try:
        print("\nMapping thicknesses to mesh vertices...")
        # Map ray intersection points to closest mesh vertices for visualization
        tree = KDTree(full_locations_mm)
        distances, indices = tree.query(mesh.vertices * 1000.0, k=1)

        vertex_thicknesses_mm = np.zeros(len(mesh.vertices))
        valid_dist_mask = distances < 1.0 
        vertex_thicknesses_mm[valid_dist_mask] = full_thicknesses_mm[indices[valid_dist_mask]]

        faces_pv = np.hstack((np.full((mesh.faces.shape[0], 1), 3), mesh.faces)).flatten()
        pv_mesh = pyvista.PolyData(mesh.vertices * 1000.0, faces_pv)
        pv_mesh.point_data['Thickness (mm)'] = vertex_thicknesses_mm 

        plotter = pyvista.Plotter(off_screen=off_screen)
        
        sargs = dict(
            title_font_size=18,
            label_font_size=14,
            n_labels=7,
            italic=False,
            fmt="%.2f",
            color='black',
            title='Thickness (mm)',
        )

        plotter.add_mesh(
            pv_mesh, 
            scalars='Thickness (mm)', 
            cmap='viridis',
            clim=[vmin, vmax], 
            scalar_bar_args=sargs,
            show_edges=False,
            smooth_shading=True,
            show_scalar_bar=False
        )

        plotter.add_scalar_bar(
            title='Thickness (mm)',
            vertical=True,
            #position_x=0.70,   #wire
            position_x=0.75,
            position_y=0.05,
            width=0.08,
            height=0.9,
            bold=True,
            label_font_size=14,
            title_font_size=18,
            italic=False,
            fmt="%.2f"
        )

        # Draw the ROI wireframe bounding box
        roi_box = pyvista.Box(bounds=list(roi_bounds))
        plotter.add_mesh(roi_box, color='red', style='wireframe', line_width=3, opacity=1.0, name='ROI_Box')

        plotter.view_vector([0.75, 0.75, 1.0])     
        #plotter.background_color = 'white'
        grid_settings = {
            'color': 'gray',
            'font_size': 16,
            'bold': True,
            'fmt': "%.1f",
        }

        plotter.show_grid(**grid_settings)
        
        if off_screen:
            plotter.screenshot(output_image_path, transparent_background=True, window_size=[1920, 1080])
            
            svg_path = output_image_path.replace('.png', '.svg')
            plotter.save_graphic(svg_path)
            
            #print(f"PyVista 3D render saved as PNG to: {output_image_path}")
            print(f"PyVista 3D render saved as SVG to: {svg_path}")
        else:
            plotter.show()
            

    except Exception as e:
        print(f"Error during PyVista rendering: {e}")
    finally:
        if 'plotter' in locals(): plotter.close()

# =============================================================================
# === MAIN ===
# =============================================================================
# %%

if __name__ == "__main__":
    if not os.path.exists(INPUT_STL):
        print(f"Warning: STL file not found at '{INPUT_STL}'.")
    else:
        target_sigma_range = MATERIAL_RANGES.get(SAMPLE_CHOICE, (1.0, 1.0))
        
        # Generate full and ROI thickness arrays
        mesh, full_locations_mm, full_thicknesses_mm, roi_thicknesses_mm, roi_bounds_mm = generate_thickness_maps(
            INPUT_STL, resolution=RAY_RESOLUTION
        )
        
        if roi_thicknesses_mm is not None:
            # Extract transmission from ROOT files and calculate macroscopic cross-section
            calculate_simulation_results(OPEN_BEAM_FILE, RESULTS_FILE, roi_thicknesses_mm, SAMPLE_CHOICE, target_sigma_range)
            
            #Full object limits
            vmin_mm, vmax_mm = np.min(full_thicknesses_mm), np.max(full_thicknesses_mm)
            #ROI-specific limits
            min_val=np.min(roi_thicknesses_mm)
            max_val=np.max(roi_thicknesses_mm)

            full_bounds_mm = (mesh.bounds[0] * 1000.0, mesh.bounds[1] * 1000.0)
            plot_roi_projections(full_locations_mm, full_thicknesses_mm, full_bounds_mm, roi_bounds_mm, OUTPUT_PREFIX, vmin_mm, vmax_mm)
            plot_thickness_statistics(roi_thicknesses_mm, OUTPUT_PREFIX, np.min(roi_thicknesses_mm), np.max(roi_thicknesses_mm), "ROI")

            # 4. Render PyVista 3D Plot using the same roi_bounds_mm
            render_3d_thickness_map(
                mesh, 
                full_locations_mm,
                full_thicknesses_mm, 
                roi_bounds_mm,
                vmin_mm, 
                vmax_mm, 
                OUTPUT_PREFIX, 
                off_screen=True
            )
# %% 