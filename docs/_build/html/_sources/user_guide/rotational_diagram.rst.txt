==================
Rotational Diagram
==================

Using the Rotational Diagram Tool
==================================

**Accessing the Tool**

1. Click the **Rot Diag** button in the main toolbar
2. The Rotational Diagram window opens

**Step 1: Fetch HITRAN Data**

1. Set wavelength range (μm):
   
   - **Min Wavelength**: Lower bound of wavelength range
   - **Max Wavelength**: Upper bound of wavelength range

2. Select H₂ isotopologue (use **1** for main isotope)

3. (Optional) Filter by vibrational levels:
   
   - Check **"Include all transitions"** to get all v' → v'' transitions
   - Uncheck to specify specific v_upper and v_lower values

4. Click **Fetch HITRAN Lines**

   - A background thread fetches molecular data from the HITRAN database
   - Progress shown during fetch
   - Parsed lines appear in the table below

**Step 2: Select Lines and Enter Data**

The table displays all fetched H₂ transitions with columns:

- **☑**: Checkbox to select/deselect each line
- **λ (μm)**: Wavelength
- **Transition**: v' J' → v'' J'' notation
- **A_ul (s⁻¹)**: Einstein coefficient
- **E_u (K)**: Upper state energy
- **g_u**: Statistical weight
- **Flux**: Observed line flux (your data - enter here)
- **Flux Error**: Uncertainty in flux
- **Error %**: Percentage uncertainty (auto-calculated)

**Entering Flux Values:**

1. Check the boxes for lines you want to include in the fit
2. Enter observed line fluxes in the **Flux** column
3. Enter flux errors in either:
   
   - **Flux Error** column (absolute values), OR
   - **Error %** column (percentage uncertainties)
   
   Use the **Error Mode** toggle to switch between absolute values and percentages.
   The columns auto-calculate between the two formats as you type.

**Step 3: Input Physical Parameters**

At the bottom of the window, provide:

- **R (AU)**: Emitting source radius in astronomical units (e.g., disk radius)
- **Distance (pc)**: Distance to source in parsecs
- **Fit Mode**: 
  
  - **1T**: Single-temperature fit (assumes one thermal component)
  - **2T**: Two-temperature fit (models warm and hot components)

**Step 4: Run the Fit**

1. Click **Run Rotational Fit** button
2. The tool performs the analysis:
   
   - Computes column densities from fluxes
   - Fits 1-temperature and/or 2-temperature models
   - Calculates rotational temperatures and uncertainties
   - Generates visualization

**Step 5: Interpret Results**

A dialog displays both text results and a plot:

**Text Results:**

For **1T fit**:
  - **T** (K): Rotational temperature with uncertainty
  - **Ncol** (cm⁻²): Total column density from intercept

For **2T fit**:
  - **T₁, T₂** (K): Warm and hot component temperatures
  - **N₁, N₂** (cm⁻²): Column densities for each component

**Plot Visualization:**

- **Black dots with error bars**: Observed data (ln(N_u/g_u) vs E_u)
- **Dashed blue line**: Warm component (2T only)
- **Dashed orange line**: Hot component (2T only)
- **Solid red line**: Combined 2-temperature model fit (2T only)
- **Dashed blue line**: 1T model fit (1T mode only)

Use the toolbar to **zoom**, **pan**, and **save** the plot.
