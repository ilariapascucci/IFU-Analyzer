=========
Colormaps
=========

Available colormaps for visualizing intensity maps.

.. list-table:: Available Colormaps
   :widths: 20 30 50
   :header-rows: 1

   * - Name
     - Type
     - Best Use
   * - Viridis
     - Perceptually Uniform
     - Default, scientific papers, colorblind-friendly
   * - Plasma
     - Perceptually Uniform
     - High contrast, presentations
   * - Inferno
     - Perceptually Uniform
     - Dark backgrounds, dramatic effect
   * - Magma
     - Perceptually Uniform
     - Publications, presentations
   * - Cividis
     - Colorblind-Friendly
     - Colorblind presentations
   * - Hot
     - Sequential
     - Thermal data, warm sources
   * - Cool
     - Sequential
     - Cool sources, absorbed regions
   * - RdYlBu
     - Diverging
     - Positive/negative data
   * - RdYlGn
     - Diverging
     - Positive/negative data
   * - Spectral
     - Diverging
     - Balanced positive/negative
   * - Binary
     - Sequential
     - Monochrome, printing
   * - Grey/Gray
     - Sequential
     - Printing, presentations
   * - PuBu
     - Sequential
     - Cool sources
   * - YlOrRd
     - Sequential
     - Warm sources

Choosing the Right Colormap
============================

**For Scientific Publications**

Use perceptually uniform colormaps:

- ✓ Viridis (most common)
- ✓ Plasma
- ✓ Inferno
- ✗ Avoid: Jet, Rainbow (non-uniform, misleading)

**For Presentations**

Use high-contrast colormaps:

- ✓ Hot
- ✓ Cool
- ✓ Plasma
- ✓ Inferno

**For Colorblind Accessibility**

Use colorblind-friendly colormaps:

- ✓ Viridis
- ✓ Cividis
- ✓ Greys
- ✗ Avoid: Red-Green combinations

**For Diverging Data**

Use diverging colormaps:

- ✓ RdYlBu (red, yellow, blue)
- ✓ RdYlGn (red, yellow, green)
- ✓ Spectral
- ✓ Cool/Warm diverging

**For Printing**

Use greyscale or monochrome:

- ✓ Binary
- ✓ Grey/Gray
- ✓ Bone
- ✓ Copper

Colormap Parameters
===================

**Reversing a Colormap**

Invert color order:

1. Select colormap
2. Check **Invert** checkbox
3. Colors reverse (e.g., Viridis: yellow to purple)
4. Useful for comparison or preference

**Modifying Colormap Intensity**

Adjust colormap brightness/contrast:

1. Colormap dialog → **Intensity**
2. Increase: Makes colors more vivid
3. Decrease: Makes colors more subtle
4. Useful for display preference

**Custom Colormaps**

Create your own colormap:

1. Tools → Preferences → Colormaps
2. **Create Custom**
3. Define color at points (e.g., dark at low, bright at high)
4. Save as custom colormap

Colormap Examples
=================

**Viridis (Default)**

- Purple → Green → Yellow
- Perceptually uniform
- Works well for all data types
- Colorblind-friendly
- Most commonly used

**Hot**

- Black → Red → Yellow → White
- Good for thermal data
- Intuitive for temperature
- High contrast
- Shows detail in bright regions

**Cool**

- Black → Cyan → White
- Good for cool/absorbed regions
- Opposite of Hot
- High contrast
- Complements Hot colormap

**RdYlBu (Red-Yellow-Blue)**

- Red (high) → Yellow (middle) → Blue (low)
- For diverging data
- Positive vs. negative
- Good for velocity maps (red = receding, blue = approaching)

**Grey (Grayscale)**

- Black → White
- Best for printing
- No color information loss
- Works in monochrome display

Using Colormaps Effectively
============================

**Emphasizing Features**

- Use **Hot** for bright objects on dark background
- Use **Cool** for faint features
- Use **Viridis** for balanced visibility

**Avoiding Misleading Representation**

- ✓ Use uniform colormaps (Viridis, Plasma)
- ✗ Avoid Jet/Rainbow (non-uniform, artifacts)
- ✓ Match colormap to data type (thermal, velocity, etc.)

**Comparing Multiple Wavelengths**

- Use same colormap for all wavelengths
- Aids visual comparison
- Use intensity scaling (vmin/vmax) to emphasize features

**Creating Composite Images**

RGB composites:

1. Select three wavelengths
2. Assign to R, G, B channels
3. Create 3-color image
4. Customize each channel's colormap

Colormap Resources
==================

For more colormaps:

- **Matplotlib**: Extensive library built-in
- **Colorcet**: Curated colormaps collection
- **Scientific Color Maps**: Perceptually accurate colormaps
- **Cmocean**: Oceanography-themed colormaps

Advanced: Custom Colormaps
===========================

Creating scientific colormaps in Python:

.. code-block:: python

   import matplotlib.colors as mcolors
   import numpy as np
   
   # Create custom colormap
   colors = ['darkblue', 'cyan', 'yellow', 'red']
   n_bins = 100
   cmap = mcolors.LinearSegmentedColormap.from_list(
       'custom', colors, N=n_bins
   )

Exporting Figures with Colormaps
=================================

When exporting visualizations:

1. File → Export Visualization
2. Choose format (PNG, PDF, etc.)
3 Colormap automatically included
4. For printing: Consider grayscale alternative
5. For presentations: Consider projector colorblindness

Best Practices
==============

1. **Consistency**: Use same colormap across related figures
2. **Accessibility**: Avoid red-green combinations
3. **Publication**: Use perceptually uniform colormaps
4. **Clarity**: Match colormap to data properties
5. **Documentation**: Note which colormap you used
6. **Alternatives**: Provide colorblind-friendly versions

Common Mistakes to Avoid
========================

- ❌ Using Jet colormap (non-uniform, artifacts)
- ❌ Using red-green for colorblind readers
- ❌ Changing colormaps between related figures
- ❌ Using colormaps that are visually confusing
- ✓ Test on different displays/projectors
- ✓ Include colorbar in all figures
- ✓ Consider print version early

References
==========

For deeper understanding of colormaps:

- "How Bad Is Your Colormap?" (Nuñez et al., 2018)
- Matplotlib colormap selection guide
- Scientific Python colormap libraries

Next Steps
==========

Now that you understand colormaps, see:

- :doc:`/user_guide/visualization` for display options
- :doc:`/user_guide/exporting_data` for saving figures
