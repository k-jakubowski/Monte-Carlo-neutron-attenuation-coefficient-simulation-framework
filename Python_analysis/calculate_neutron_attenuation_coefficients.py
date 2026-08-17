import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import math

# =============================================================================
# === USER INPUT: SELECT YOUR SAMPLE HERE ===
# =============================================================================
SAMPLE_CHOICE = "solder"

if SAMPLE_CHOICE == "solder":
    OPEN_BEAM_FILE = "/path_to/open_beam_solder.root"
    RESULTS_FILE = "/path_to/solder_sample.root"

elif SAMPLE_CHOICE == "wire":
    OPEN_BEAM_FILE = "/path_to/open_beam_wire.root"
    RESULTS_FILE = "/path_to/wire_sample.root"

elif SAMPLE_CHOICE == "gilding":
    OPEN_BEAM_FILE = "/path_to/open_beam_wire.root"
    RESULTS_FILE = "/path_to/wire_sample.root"

# =============================================================================

# === EXPERIMENTAL CONSTANTS ===
# Define the plotting limits in MeV
X_AXIS_MIN_MeV = 1e-9 
X_AXIS_MAX_MeV = 1e1

# --- Sample Thicknesses and Data Loading ---

sample_thicknesses = {
    "wire": 0.04121,
    "solder": 0.08664,
    #"solder": 0.04703,
    "gilding": 0.01635
}

def load_cross_section_data(filename="/path_to//Gd_cross_sec.txt"):
    """Loads Energy (eV) and Cross-Section (barns) from the external file."""
    try:
        # Load the first two columns (E, Sig), skipping lines starting with '[' (comments/source info)
        cross_sec_data = np.loadtxt(filename, usecols=(0, 1), comments='[')
        return cross_sec_data
    except Exception as e:
        print(f"ERROR: Could not load cross-section data from {filename}. Check file name and format.")
        print(f"Error details: {e}")
        return None

def get_gd_efficiency_interpolator(gd157_enrichment_factor, optical_efficiency_factor, num_bins=100):
    """
    Creates an interpolator for the normalized Gd-157 detector efficiency 
    using the *true* cross-section data from Gd_cross_sec.txt.
    
    The detector efficiency (epsilon) is proportional to the true cross-section (sigma_capt),
    the Gd-157 enrichment, and the optical efficiency.
    epsilon = sigma_capt * gd157_enrichment_factor * optical_efficiency_factor
    """
    # 1. Load External Cross-Section Data
    cross_sec_data = load_cross_section_data(filename="/path_to/Gd_cross_sec.txt")
    if cross_sec_data is None or cross_sec_data.size == 0:
        # Returning None will cause the main execution to skip the calculation/plotting
        return None, None, None, None, None

    cross_sec_E_eV = cross_sec_data[:, 0]
    # This is the true cross-section in barns, used for plotting.
    cross_section_barns = cross_sec_data[:, 1] 

    # 2. Calculate the Raw Detector Efficiency (epsilon_raw)
    epsilon_raw = cross_section_barns * gd157_enrichment_factor * optical_efficiency_factor
    
    # 3. Normalize the Efficiency
    normalized_efficiency = epsilon_raw / epsilon_raw.max()

    # 4. Create the interpolator function
    efficiency_func = interp1d(cross_sec_E_eV, normalized_efficiency, 
                               kind='linear', bounds_error=False, 
                               fill_value=(normalized_efficiency[0], 0.0))

    # 5. Define the logarithmic bins and plot limits (in MeV)
    log_bins_mev = np.logspace(np.log10(X_AXIS_MIN_MeV), np.log10(X_AXIS_MAX_MeV), num_bins)

    # 6. Convert Energy axis to MeV for plotting (the Interpolator uses eV)
    eff_energy_mev = cross_sec_E_eV * 1e-6 
    
    # Return efficiency function, energy points (in MeV), normalized efficiency, cross-section, and log bins
    return (efficiency_func, eff_energy_mev, normalized_efficiency, cross_section_barns, log_bins_mev)

# --- Macroscopic Coefficient Calculation (Remains the same) ---

def calculate_macroscopic_coefficient(sample_choice, efficiency_func):
    """
    Calculate the macroscopic attenuation coefficient (Sigma) using the ratio 
    of the neutron counts and the sample thickness (L).
    
    Calculate both:
    1. Uncorrected Sigma: Uses raw (unweighted) neutron counts.
    2. Corrected Sigma: Uses efficiency-weighted neutron counts.
    
    ln(I_open / I_sample) = Sigma * L 
    """
    
    sample_thickness = sample_thicknesses[sample_choice]

    # --- Load Data from ROOT files ---
    try:
        # Load the neutron energy data from the ROOT files
        open_file = uproot.open(OPEN_BEAM_FILE)
        results_file = uproot.open(RESULTS_FILE)
        
        # Access the branch using the full path and call .array() on the branch object itself
        # [Tree_Name][Branch_Name].array()
        energies_open_mev = open_file["NeutronEnergyData/NeutronKE_MeV"].array()
        energies_results_mev = results_file["NeutronEnergyData/NeutronKE_MeV"].array()

    except Exception as e:
        # Use the enhanced error message as recommended in the previous step
        print(f"CRITICAL ERROR LOADING ROOT FILES: {e}") 
        print(f"Check if files exist: Open='{OPEN_BEAM_FILE}', Sample='{RESULTS_FILE}'")
        print(f"Warning: Could not open ROOT files. Proceeding with mock data for flux/efficiency visualization only.")
        energies_open_mev = np.array([])
        energies_results_mev = np.array([])
    
    # =========================================================================
    # === UNCORRECTED Macroscopic Coefficient (Raw Counts) - NO FACTORS USED ===
    # =========================================================================
    
    # I_open_raw and I_results_raw are simply the total number of detected neutrons
    I_open_raw = len(energies_open_mev)
    I_results_raw = len(energies_results_mev)
    
    calculated_sigma_uncorrected = np.nan
    if I_results_raw > 0 and I_open_raw > 0:
        try:
            # Sigma_uncorrected = (1/L) * ln(I_open_raw / I_results_raw)
            calculated_sigma_uncorrected = (1.0 / sample_thickness) * np.log(I_open_raw / I_results_raw)
        except RuntimeWarning:
            print("Warning: Division by zero or log of zero occurred for uncorrected sigma.")

    # =========================================================================
    # === CORRECTED Macroscopic Coefficient (Efficiency Weighted) - FACTORS USED ===
    # =========================================================================

    # --- Calculate Weighted Counts (Numerators) ---
    E_open_eV = energies_open_mev * 1e6
    E_results_eV = energies_results_mev * 1e6

    # Calculate the normalized efficiency weight for each detected neutron
    efficiency_weights_open = efficiency_func(E_open_eV)
    efficiency_weights_results = efficiency_func(E_results_eV)
    
    I_open = np.sum(efficiency_weights_open)
    I_results = np.sum(efficiency_weights_results)
    
    # --- Calculate the Corrected Macroscopic Attenuation Coefficient ---
    calculated_sigma_corrected = np.nan
    if I_results > 0 and I_open > 0:
        try:
            # Sigma_corrected = (1/L) * ln(I_open / I_results)
            calculated_sigma_corrected = (1.0 / sample_thickness) * np.log(I_open / I_results)
            
        except RuntimeWarning:
            print("Warning: Division by zero or log of zero occurred for corrected sigma.")
    else:
        print("Warning: Weighted counts are zero. Cannot calculate corrected coefficient.")

    print("\n----------------------------------------------------")
    print(f"Sample: {sample_choice.capitalize()} (Thickness L: {sample_thickness:.4f} cm)")
    print("----------------------------------------------------")
    
    # Uncorrected Results
    print("--- UNCORRECTED COEFFICIENT (Raw Counts) ---")
    print(f"Raw Counts - Open Beam (I_open_raw): {I_open_raw:,d}")
    print(f"Raw Counts - Sample (I_results_raw): {I_results_raw:,d}")
    print(f"Macroscopic Attenuation Coefficient (Sigma_uncorrected): {calculated_sigma_uncorrected:,.4f} cm⁻¹")
    print("----------------------------------------------------")

    # Corrected Results
    print("--- CORRECTED COEFFICIENT (Efficiency Weighted) ---")
    print(f"Weighted Counts - Open Beam (I_open_corrected): {I_open:,.4e}")
    print(f"Weighted Counts - Sample (I_results_corrected): {I_results:,.4e}")
    print(f"Macroscopic Attenuation Coefficient (Sigma_corrected): {calculated_sigma_corrected:,.4f} cm⁻¹")
    print("----------------------------------------------------")

    # --- Plotting ---
    
    # Using dummy factors 1.0, 1.0 for the plot data, as the factors cancel out during normalization
    # and only the normalized cross-section shape is needed for plotting.
    gd_efficiency_func_results = get_gd_efficiency_interpolator(1.0, 1.0)

    if gd_efficiency_func_results[0] is None:
        return calculated_sigma_corrected, calculated_sigma_uncorrected, np.nan, np.nan, "ERROR: Plotting skipped due to data loading failure."

    eff_energy_mev = gd_efficiency_func_results[1]
    eff_sigma = gd_efficiency_func_results[2]
    cross_section_barns = gd_efficiency_func_results[3]
    log_bins_mev = gd_efficiency_func_results[4]
    
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # --- Top Plot: Efficiency and Cross-Section ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eff_energy_mev, eff_sigma, 'r-', label='Gd-157 Capture Efficiency (Normalized)', linewidth=2)
    ax1.set_ylabel('Normalized Detector Efficiency (0 to 1)')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(X_AXIS_MIN_MeV, X_AXIS_MAX_MeV)
    ax1.grid(True, which='both', linestyle='--')
    
    ax2 = ax1.twinx()
    # Plot the true cross-section data
    ax2.plot(eff_energy_mev, cross_section_barns, 'b--', label='Gd-157 $\sigma_{capt}$ (True Data)', alpha=0.6)
    ax2.set_ylabel('Capture Cross-Section (barns)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_yscale('log')
    ax2.set_ylim(max(1e0, cross_section_barns.min() * 0.5), cross_section_barns.max() * 2) 
    
    # --- Bottom Plot: Neutron Spectra (Counts) ---
    ax4 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plotting the Neutron Spectra (MeV vs. MeV bins)
    if len(energies_open_mev) > 0 and len(energies_results_mev) > 0:
        #ax4.hist(energies_open_mev, bins=log_bins_mev, histtype='step', lw=2, color='red', label=f'Open Beam (N={len(energies_open_mev)})')
        #ax4.hist(energies_results_mev, bins=log_bins_mev, histtype='step', lw=2, color='black', label=f'Results Beam (N={len(energies_results_mev)})')
        ax4.hist(energies_open_mev, bins=log_bins_mev, histtype='step', color='black', lw=2, label=f'Incident neutron spectrum')
        ax4.hist(energies_results_mev, bins=log_bins_mev, histtype='step', color='gold', lw=2, label=f'Attenuated neutron spectrum')

        ax4.set_ylabel('Neutron Counts per Log Bin')
        ax4.set_yscale('log')
        ax4.set_ylim(1, ax4.get_ylim()[1] * 10) 
    else:
        # Provide a warning if data is missing
        print("Warning: ROOT data arrays were empty, showing mock count axis.")
        ax4.set_ylabel('Neutron Counts (Data Missing)')
        ax4.set_yscale('log')

    # Top Plot Legend: Efficiency and Cross-Section
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    # Bottom Plot Legend: Open/Results Beam Counts
    if len(energies_open_mev) > 0 and len(energies_results_mev) > 0:
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax4.legend(lines4, labels4, loc='upper right')

    ax4.set_xlabel('Neutron Energy (MeV)') 
    ax1.set_title(f'Neutron Attenuation Analysis for {sample_choice.capitalize()} (Thickness: {sample_thickness:.4f} cm)')
    plt.tight_layout()
    
    PLOT_FILENAME = "/path_to/neutron_attenuation_analysis.png"
    plt.savefig(PLOT_FILENAME)
    plt.close()

# =========================================================================
    # === NEW ADDITION: Separate Spectra Plot with Grid ===
    # =========================================================================
    SPECTRA_PLOT_FILENAME = "/path_to/neutron_spectra_only.png"
    
    # Create a new, separate figure
    fig_spec = plt.figure(figsize=(10, 6))
    ax_spec = fig_spec.add_subplot(111)

    if len(energies_open_mev) > 0 and len(energies_results_mev) > 0:
        # Plot histograms using the same log bins
        ax_spec.hist(energies_open_mev, bins=log_bins_mev, histtype='step', color='black', lw=2, label='Incident neutron spectrum')
        ax_spec.hist(energies_results_mev, bins=log_bins_mev, histtype='step', color='gold', lw=2, label='Attenuated neutron spectrum')
        
        # Formatting
        ax_spec.set_xlabel('Neutron Energy (MeV)')
        ax_spec.set_ylabel('Neutron Counts per Log Bin')
        ax_spec.set_title(f'Neutron Spectra for {sample_choice.capitalize()}')
        ax_spec.set_yscale('log')
        ax_spec.set_xscale('log') # Explicitly set x-axis to log
        ax_spec.set_xlim(X_AXIS_MIN_MeV, X_AXIS_MAX_MeV) # Match the limits``
        
        # Add the requested Grid
        ax_spec.grid(True, which='both', linestyle='--', alpha=0.7)
        
        ax_spec.legend(loc='upper right')
    else:
        ax_spec.text(0.5, 0.5, "No Data Loaded", ha='center', va='center', transform=ax_spec.transAxes)

    plt.tight_layout()
    plt.savefig(SPECTRA_PLOT_FILENAME)
    plt.close()
    
    print(f"Separate neutron spectra plot saved to {SPECTRA_PLOT_FILENAME}")
    # =========================================================================

    return calculated_sigma_corrected, calculated_sigma_uncorrected, eff_energy_mev, cross_section_barns, PLOT_FILENAME

# --- Main Execution ---
# Check if running in a main context (prevents execution on import)
if __name__ == "__main__":
    # Call the function once and unpack the results
    GD157_ENRICHMENT_FACTOR = 0.8817 # enrichment in the screen
    OPTICAL_EFFICIENCY_FACTOR = 0.70 # Light output above 70%

    # Call the function once and unpack the results
    # MODIFIED: Pass factors to the interpolator
    efficiency_interp_func, eff_energy_mev, normalized_efficiency, cross_section_barns, log_bins_mev = \
        get_gd_efficiency_interpolator(GD157_ENRICHMENT_FACTOR, OPTICAL_EFFICIENCY_FACTOR)    
    # Check for successful loading before proceeding
    if efficiency_interp_func is None:
        print("Error: Detector efficiency data could not be loaded. Exiting.")
    else:
        # Pass only the interpolator function to the calculation function
        calculated_sigma_corrected, calculated_sigma_uncorrected, eff_E_mev, eff_sigma_barns, plot_filename = \
            calculate_macroscopic_coefficient(SAMPLE_CHOICE, efficiency_interp_func)
        
        if plot_filename and not isinstance(plot_filename, str) and not math.isnan(calculated_sigma_corrected):
            # The result of the plot saving (the filename) is returned for rendering.
            print(f"\nFinal plot saved to {plot_filename}")