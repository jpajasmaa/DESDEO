<script lang="ts">
	/**
	 * ParallelCoordinates.svelte
	 * --------------------------------
	 * Responsive parallel coordinates plot component using D3.
	 *
	 * @author Giomara Larraga <glarragw@jyu.fi>
	 * @created June 2025
	 *
	 * @description
	 * Renders a responsive parallel coordinates plot using D3.js.
	 * Each line represents a solution/data point, and each vertical axis represents a dimension/objective.
	 * Supports additional reference information like reference points, preferred ranges, and preferred solutions.
	 * Features line selection and axis brushing for filtering.
	 *
	 * @props
	 * - data: Array<{ [key: string]: number }> — Array of data points where each object has values for each dimension
	 * - dimensions: Array<{ name: string; min?: number; max?: number; direction?: 'max' | 'min' }> — Dimension definitions
	 * - referenceData?: {
	 *     referencePoint?: { [key: string]: number }; // Reference point values for each dimension
	 *     preferredRanges?: { [key: string]: { min: number; max: number } }; // Preferred ranges for each dimension
	 *     preferredSolutions?: Array<{ [key: string]: number }>; // Array of preferred solutions
	 *     nonPreferredSolutions?: Array<{ [key: string]: number }>; // Array of non-preferred solutions
	 *   }
	 * - options: {
	 *     showAxisLabels: boolean; // show dimension names
	 *     highlightOnHover: boolean; // highlight lines on hover
	 *     strokeWidth: number; // line thickness
	 *     opacity: number; // line opacity
	 *     enableBrushing: boolean; // enable axis brushing for filtering
	 *   }
	 * - selectedIndex: number | null — index of selected line (only one at a time)
	 * - brushFilters: { [dimension: string]: [number, number] } — brush filter ranges for each dimension
	 * - onLineSelect?: (index: number | null, data: any | null) => void — callback when line is selected/deselected
	 * - onBrushFilter?: (filters: { [dimension: string]: [number, number] }) => void — callback when brush filters change
	 *
	 * @features
	 * - Interactive line highlighting on hover
	 * - Single line selection on click
	 * - Axis brushing for range filtering
	 * - Custom color palette for different data points
	 * - Responsive to container size (ResizeObserver)
	 * - Customizable axis ranges and directions
	 * - Reference point visualization (dashed line)
	 * - Preferred ranges visualization (colored bands)
	 * - Preferred/non-preferred solutions (different line styles)
	 */

	// --- Import required libraries ---
	import { onMount, onDestroy } from 'svelte';
	import * as d3 from 'd3'; // D3.js for data visualization
	import { COLOR_PALETTE } from '../utils/colors'; // Custom color palette for styling

	// --- Type Definitions ---
	/**
	 * Type definition for reference data structure
	 * Contains optional reference information for enhanced visualization
	 */
	type ReferenceData = {
		referencePoint?: { [key: string]: number }; // Single reference point across all dimensions
		preferredRanges?: { [key: string]: { min: number; max: number } }; // Preferred value ranges per dimension
		preferredSolutions?: Array<{ [key: string]: number }>; // Array of preferred solution points
		nonPreferredSolutions?: Array<{ [key: string]: number }>; // Array of non-preferred solution points
	};

	// --- Component Props ---
	// Main data array - each object represents one solution/data point
	export let data: { [key: string]: number }[] = [];

	// Dimension definitions - describes each axis with optional constraints
	export let dimensions: { name: string; min?: number; max?: number; direction?: 'max' | 'min' }[] =
		[];

	// Optional reference data for enhanced visualization
	export let referenceData: ReferenceData | undefined = undefined;

	// Chart configuration options
	export let options: {
		showAxisLabels: boolean; // Whether to show dimension names above axes
		highlightOnHover: boolean; // Whether to highlight lines on mouse hover
		strokeWidth: number; // Thickness of data lines
		opacity: number; // Opacity of non-selected lines
		enableBrushing: boolean; // Whether to enable axis brushing for filtering
	} = {
		showAxisLabels: true,
		highlightOnHover: true,
		strokeWidth: 2,
		opacity: 0.7,
		enableBrushing: true
	};

	// Index of currently selected line (null = no selection)
	export let selectedIndex: number | null = null;

	// Active brush filters - maps dimension name to [y1, y2] pixel coordinates
	export let brushFilters: { [dimension: string]: [number, number] } = {};

	// Callback functions for parent component communication
	export let onLineSelect: ((index: number | null, data: any | null) => void) | undefined =
		undefined;
	export let onBrushFilter:
		| ((filters: { [dimension: string]: [number, number] }) => void)
		| undefined = undefined;

	// --- Internal State Variables ---
	let width = 500; // Current container width in pixels
	let height = 400; // Current container height in pixels
	let svg: SVGSVGElement; // Reference to the SVG element
	let container: HTMLDivElement; // Reference to the container div
	let resizeObserver: ResizeObserver; // Observer for container size changes
	let brushes: { [dimension: string]: d3.BrushBehavior<unknown> } = {}; // D3 brush objects per dimension
	let scales: { [key: string]: d3.ScaleLinear<number, number> } = {}; // D3 scales for each dimension

	/**
	 * Creates linear scales for each dimension
	 * Scales map data values to pixel coordinates on the y-axis
	 *
	 * @param innerHeight - Available height for the chart area
	 * @param margin - Margin object with top/bottom spacing
	 * @returns Object mapping dimension names to D3 linear scales
	 */
	function createScales(innerHeight: number, margin: { top: number; bottom: number }) {
		const newScales: { [key: string]: d3.ScaleLinear<number, number> } = {};

		dimensions.forEach((dim) => {
			// Extract all values for this dimension from the dataset
			const values = data.map((d) => d[dim.name]).filter((v) => v !== undefined && v !== null);

			// Determine the domain (data range) for this dimension
			let domain: [number, number];
			if (dim.min !== undefined && dim.max !== undefined) {
				// Use predefined min/max if provided
				domain = [dim.min, dim.max];
			} else {
				// Calculate min/max from actual data
				const extent = d3.extent(values) as [number, number];
				domain = extent || [0, 1]; // Fallback to [0,1] if no data
			}

			// Reverse domain for 'max' direction to show higher values at top
			if (dim.direction === 'max') {
				domain = [domain[1], domain[0]];
			}

			// Create linear scale mapping data domain to pixel range
			newScales[dim.name] = d3
				.scaleLinear()
				.domain(domain) // Data values
				.range([innerHeight - margin.bottom, margin.top]); // Pixel coordinates (bottom to top)
		});

		// Store scales for use in other functions
		scales = newScales;
		return newScales;
	}

	/**
	 * Creates a D3 line generator for drawing parallel coordinate lines
	 * Each line connects data points across all dimensions
	 *
	 * @param scales - Scale functions for each dimension
	 * @param xScale - Scale for positioning dimensions horizontally
	 * @returns D3 line generator function
	 */
	function createLineGenerator(
		scales: { [key: string]: d3.ScaleLinear<number, number> },
		xScale: d3.ScalePoint<string>
	) {
		return d3
			.line<[string, number]>() // Line generator for [dimension, value] tuples
			.x(([dimension]) => xScale(dimension)!) // X position based on dimension
			.y(([dimension, value]) => scales[dimension](value)) // Y position based on scaled value
			.curve(d3.curveLinear); // Use straight lines between points
	}

	/**
	 * Checks if a data point passes all active brush filters
	 * Used to determine which lines should be visible
	 *
	 * @param dataPoint - Single data point to test
	 * @returns true if the point passes all filters, false otherwise
	 */
	function passesFilters(dataPoint: { [key: string]: number }): boolean {
		// Check each active brush filter
		for (const [dimension, [min, max]] of Object.entries(brushFilters)) {
			const value = dataPoint[dimension];
			if (value === undefined || value === null) continue; // Skip missing values

			// Get the scale for this dimension
			const scale = scales[dimension];
			if (!scale) continue; // Skip if no scale available

			// Convert brush pixel coordinates back to data values
			const dataMin = scale.invert(max); // Note: inverted because y-axis goes bottom-to-top
			const dataMax = scale.invert(min);

			// Check if data value falls within the brush range
			if (value < dataMin || value > dataMax) {
				return false; // Point is outside this filter
			}
		}
		return true; // Point passes all filters
	}

	/**
	 * Updates the visibility and styling of all data lines
	 * Handles filtering, selection highlighting, and opacity
	 *
	 * @param lines - D3 selection of path elements representing data lines
	 */
	function updateLineVisibility(
		lines: d3.Selection<SVGPathElement, { [key: string]: number }, SVGGElement, unknown>
	) {
		lines
			// Hide/show lines based on brush filters
			.style('display', (d, i) => {
				const passes = passesFilters(d);
				return passes ? null : 'none'; // null = visible, 'none' = hidden
			})
			// Set opacity - selected line is fully opaque, others use default opacity
			.attr('opacity', (d, i) => {
				const passes = passesFilters(d);
				if (!passes) return 0; // Hidden lines have 0 opacity

				if (i === selectedIndex) return 1; // Selected line is fully opaque
				return options.opacity; // Other lines use configured opacity
			})
			// Set stroke color - selected line gets theme color, others are gray
			.attr('stroke', (d, i) => {
				const passes = passesFilters(d);
				if (!passes) return '#ccc'; // Hidden lines are gray

				if (i === selectedIndex) return '#3b82f6'; // Selected line uses primary color
				return '#ccc'; // Non-selected lines are gray
			})
			// Set stroke width - selected line is slightly thicker
			.attr('stroke-width', (d, i) => {
				if (i === selectedIndex) return options.strokeWidth + 1; // Selected line is thicker
				return options.strokeWidth; // Normal thickness for others
			});
	}

	/**
	 * Sets up brushing interaction for a single axis
	 * Allows users to drag vertically to filter data within a range
	 *
	 * @param svgElement - Parent SVG group element
	 * @param dimension - Name of the dimension this brush controls
	 * @param xPos - X coordinate of the axis
	 * @param innerHeight - Height of the chart area
	 * @param lines - D3 selection of data lines to update when brushing
	 */
	function setupAxisBrushing(
		svgElement: d3.Selection<SVGGElement, unknown, null, undefined>,
		dimension: string,
		xPos: number,
		innerHeight: number,
		lines: d3.Selection<SVGPathElement, { [key: string]: number }, SVGGElement, unknown>
	) {
		if (!options.enableBrushing) return; // Skip if brushing is disabled

		// Create D3 brush behavior for vertical brushing
		const brush = d3
			.brushY() // Vertical brush only
			.extent([
				[xPos - 10, 0], // Brush area: 20px wide centered on axis
				[xPos + 10, innerHeight]
			])
			// Handle brush events during dragging (real-time feedback)
			.on('brush', function (event) {
				const brushGroup = d3.select(this.parentNode);

				if (event.selection) {
					const [y1, y2] = event.selection as [number, number];

					// Remove previous highlight rectangle
					brushGroup.select('.brush-highlight').remove();

					// Add new highlight rectangle showing brush area
					brushGroup
						.append('rect')
						.attr('class', 'brush-highlight')
						.attr('x', xPos - 15) // Slightly wider than brush for visibility
						.attr('y', y1)
						.attr('width', 30)
						.attr('height', y2 - y1)
						.attr('fill', '#4a90e2') // Blue fill
						.attr('opacity', 0.2)
						.attr('stroke', '#4a90e2') // Blue border
						.attr('stroke-width', 2)
						.attr('stroke-dasharray', '3,3') // Dashed border
						.style('pointer-events', 'none'); // Don't interfere with brushing
				}
			})
			// Handle brush end events (when user releases mouse)
			.on('end', function (event) {
				const brushGroup = d3.select(this.parentNode);

				if (!event.selection) {
					// No selection = brush was cleared
					delete brushFilters[dimension]; // Remove filter for this dimension
					brushGroup.select('.brush-highlight').remove(); // Remove highlight rectangle
				} else {
					// Brush selection exists = update filter
					const [y1, y2] = event.selection as [number, number];
					brushFilters[dimension] = [y1, y2]; // Store filter coordinates

					// Ensure highlight rectangle is present
					brushGroup.select('.brush-highlight').remove();
					brushGroup
						.append('rect')
						.attr('class', 'brush-highlight')
						.attr('x', xPos - 15)
						.attr('y', y1)
						.attr('width', 30)
						.attr('height', y2 - y1)
						.attr('fill', '#4a90e2')
						.attr('opacity', 0.2)
						.attr('stroke', '#4a90e2')
						.attr('stroke-width', 2)
						.attr('stroke-dasharray', '3,3')
						.style('pointer-events', 'none');
				}

				// Update line visibility based on new filters
				updateLineVisibility(lines);

				// Notify parent component of filter changes
				onBrushFilter?.(brushFilters);
			});

		// Store brush behavior for this dimension
		brushes[dimension] = brush;

		// Create brush group and apply brush behavior
		const brushGroup = svgElement
			.append('g')
			.attr('class', `brush brush-${dimension}`)
			.attr('transform', `translate(0, 0)`)
			.call(brush);

		// Style the brush selection area (semi-transparent blue)
		brushGroup
			.selectAll('.selection')
			.style('fill', '#4a90e2')
			.style('opacity', 0.15)
			.style('stroke', '#4a90e2')
			.style('stroke-width', 1);

		// Style the brush handles (resize handles at top/bottom)
		brushGroup
			.selectAll('.handle')
			.style('fill', '#4a90e2')
			.style('stroke', '#4a90e2')
			.style('stroke-width', 2)
			.style('cursor', 'ns-resize'); // North-south resize cursor

		// Restore existing brush if it exists in brushFilters
		if (brushFilters[dimension]) {
			const [y1, y2] = brushFilters[dimension];
			brush.move(brushGroup, [y1, y2]); // Apply saved brush selection
		}
	}

	/**
	 * Handles line selection when user clicks on a data line
	 * Implements single-selection behavior (only one line can be selected)
	 *
	 * @param index - Index of the clicked line in the data array
	 * @param dataPoint - The actual data object for the clicked line
	 */
	function handleLineClick(index: number, dataPoint: { [key: string]: number }) {
		if (selectedIndex === index) {
			// Clicking on already selected line = deselect it
			selectedIndex = null;
			onLineSelect?.(null, null);
		} else {
			// Clicking on different line = select it
			selectedIndex = index;
			onLineSelect?.(index, dataPoint);
		}
	}

	/**
	 * Draws preferred ranges as colored bands behind the axes
	 * Shows visually which value ranges are preferred for each dimension
	 *
	 * @param svgElement - Parent SVG group element
	 * @param scales - Scale functions for each dimension
	 * @param xScale - Scale for positioning dimensions horizontally
	 */
	function drawPreferredRanges(
		svgElement: d3.Selection<SVGGElement, unknown, null, undefined>,
		scales: { [key: string]: d3.ScaleLinear<number, number> },
		xScale: d3.ScalePoint<string>
	) {
		if (!referenceData?.preferredRanges) return; // Skip if no preferred ranges defined

		// Create group for all preferred range visualizations
		const rangesGroup = svgElement.append('g').attr('class', 'preferred-ranges');

		// Draw a colored band for each dimension that has preferred ranges
		Object.entries(referenceData.preferredRanges).forEach(([dimName, range]) => {
			const x = xScale(dimName); // Get x position of this dimension's axis
			if (x === undefined || !scales[dimName]) return; // Skip if position/scale not available

			// Convert data range to pixel coordinates
			const yMin = scales[dimName](range.max); // Top of range (higher values)
			const yMax = scales[dimName](range.min); // Bottom of range (lower values)

			// Draw semi-transparent rectangle showing preferred range
			rangesGroup
				.append('rect')
				.attr('class', `preferred-range-${dimName}`)
				.attr('x', x - 10) // Center on axis, 20px wide
				.attr('y', yMin)
				.attr('width', 20)
				.attr('height', yMax - yMin)
				.attr('fill', '#e6f3ff') // Light blue fill
				.attr('stroke', '#4a90e2') // Blue border
				.attr('stroke-width', 1)
				.attr('opacity', 0.3);

			// Add "preferred" label next to the range
			rangesGroup
				.append('text')
				.attr('x', x + 15) // Position to the right of the axis
				.attr('y', yMin + (yMax - yMin) / 2) // Center vertically in the range
				.attr('text-anchor', 'start')
				.style('font-size', '9px')
				.style('fill', '#4a90e2')
				.text('preferred');
		});
	}

	/**
	 * Draws the reference point as a dashed line across all dimensions
	 * Shows a specific target or comparison point
	 *
	 * @param svgElement - Parent SVG group element
	 * @param scales - Scale functions for each dimension
	 * @param xScale - Scale for positioning dimensions horizontally
	 * @param line - D3 line generator for drawing the path
	 */
	function drawReferencePoint(
		svgElement: d3.Selection<SVGGElement, unknown, null, undefined>,
		scales: { [key: string]: d3.ScaleLinear<number, number> },
		xScale: d3.ScalePoint<string>,
		line: d3.Line<[string, number]>
	) {
		if (!referenceData?.referencePoint) return; // Skip if no reference point defined

		// Create group for reference point visualization
		const referenceGroup = svgElement.append('g').attr('class', 'reference-point');

		// Convert reference point data to line format
		const refLineData: [string, number][] = dimensions
			.map((dim) => [dim.name, referenceData.referencePoint![dim.name]])
			.filter(([, value]) => value !== undefined && value !== null);

		if (refLineData.length > 0) {
			// Draw dashed line connecting reference values across all dimensions
			referenceGroup
				.append('path')
				.datum(refLineData)
				.attr('d', line) // Use line generator to create path
				.attr('fill', 'none')
				.attr('stroke', '#ff6b6b') // Red color for reference
				.attr('stroke-width', options.strokeWidth + 1) // Slightly thicker than data lines
				.attr('stroke-dasharray', '8,4') // Dashed pattern
				.attr('opacity', 0.9);

			// Add circles at each axis to highlight reference values
			refLineData.forEach(([dimName, value]) => {
				const x = xScale(dimName);
				const y = scales[dimName](value);
				if (x !== undefined && !isNaN(y)) {
					referenceGroup
						.append('circle')
						.attr('cx', x)
						.attr('cy', y)
						.attr('r', 4) // Small circle radius
						.attr('fill', '#ff6b6b') // Same red as line
						.attr('stroke', '#fff') // White border for visibility
						.attr('stroke-width', 2);
				}
			});

			// Note: Reference point label is commented out to reduce visual clutter
			/*referenceGroup
                .append('text')
                .attr('x', 10)
                .attr('y', -10)
                .style('font-size', '11px')
                .style('fill', '#ff6b6b')
                .style('font-weight', 'bold')
                .text('Reference Point');*/
		}
	}

	/**
	 * Draws preferred and non-preferred solutions with distinct visual styles
	 * Shows example solutions that are considered good or bad
	 *
	 * @param svgElement - Parent SVG group element
	 * @param scales - Scale functions for each dimension
	 * @param xScale - Scale for positioning dimensions horizontally
	 * @param line - D3 line generator for drawing paths
	 */
	function drawReferenceSolutions(
		svgElement: d3.Selection<SVGGElement, unknown, null, undefined>,
		scales: { [key: string]: d3.ScaleLinear<number, number> },
		xScale: d3.ScalePoint<string>,
		line: d3.Line<[string, number]>
	) {
		// Draw preferred solutions (good examples)
		if (referenceData?.preferredSolutions) {
			const preferredGroup = svgElement.append('g').attr('class', 'preferred-solutions');

			referenceData.preferredSolutions.forEach((solution, index) => {
				// Convert solution data to line format
				const solutionData: [string, number][] = dimensions
					.map((dim) => [dim.name, solution[dim.name]])
					.filter(([, value]) => value !== undefined && value !== null);

				if (solutionData.length > 0) {
					// Draw dashed line for preferred solution
					preferredGroup
						.append('path')
						.datum(solutionData)
						.attr('d', line)
						.attr('fill', 'none')
						.attr('stroke', '#4ecdc4') // Teal color for preferred
						.attr('stroke-width', options.strokeWidth + 2) // Thicker than normal lines
						.attr('stroke-dasharray', '4,2') // Different dash pattern
						.attr('opacity', 0.8);

					// Add triangle markers at each axis point
					solutionData.forEach(([dimName, value]) => {
						const x = xScale(dimName);
						const y = scales[dimName](value);
						if (x !== undefined && !isNaN(y)) {
							preferredGroup
								.append('polygon')
								.attr('points', `${x},${y - 4} ${x + 4},${y + 3} ${x - 4},${y + 3}`) // Triangle shape
								.attr('fill', '#4ecdc4')
								.attr('stroke', '#fff')
								.attr('stroke-width', 1);
						}
					});
				}
			});

			// Note: Preferred solutions label is commented out to reduce visual clutter
			/*if (referenceData.preferredSolutions.length > 0) {
                svgElement
                    .append('text')
                    .attr('x', 10)
                    .attr('y', 10)
                    .style('font-size', '11px')
                    .style('fill', '#4ecdc4')
                    .style('font-weight', 'bold')
                    .text(`${referenceData.preferredSolutions.length} Preferred Solution(s)`);
            }*/
		}

		// Draw non-preferred solutions (bad examples)
		if (referenceData?.nonPreferredSolutions) {
			const nonPreferredGroup = svgElement.append('g').attr('class', 'non-preferred-solutions');

			referenceData.nonPreferredSolutions.forEach((solution, index) => {
				// Convert solution data to line format
				const solutionData: [string, number][] = dimensions
					.map((dim) => [dim.name, solution[dim.name]])
					.filter(([, value]) => value !== undefined && value !== null);

				if (solutionData.length > 0) {
					// Draw dashed line for non-preferred solution
					nonPreferredGroup
						.append('path')
						.datum(solutionData)
						.attr('d', line)
						.attr('fill', 'none')
						.attr('stroke', '#e74c3c') // Red color for non-preferred
						.attr('stroke-width', options.strokeWidth + 1)
						.attr('stroke-dasharray', '2,3') // Dense dash pattern
						.attr('opacity', 0.6);

					// Add X markers at each axis point
					solutionData.forEach(([dimName, value]) => {
						const x = xScale(dimName);
						const y = scales[dimName](value);
						if (x !== undefined && !isNaN(y)) {
							nonPreferredGroup
								.append('path')
								.attr(
									'd',
									`M${x - 3},${y - 3} L${x + 3},${y + 3} M${x + 3},${y - 3} L${x - 3},${y + 3}` // X shape
								)
								.attr('stroke', '#e74c3c')
								.attr('stroke-width', 2);
						}
					});
				}
			});

			// Note: Non-preferred solutions label is commented out to reduce visual clutter
			/*if (referenceData.nonPreferredSolutions.length > 0) {
                svgElement
                    .append('text')
                    .attr('x', 10)
                    .attr('y', 30)
                    .style('font-size', '11px')
                    .style('fill', '#e74c3c')
                    .style('font-weight', 'bold')
                    .text(`${referenceData.nonPreferredSolutions.length} Non-Preferred Solution(s)`);
            }*/
		}
	}

	/**
	 * Main function that draws the entire parallel coordinates plot
	 * Orchestrates all the drawing functions and handles the overall layout
	 */
	function drawChart(): void {
		if (!data.length || !dimensions.length) return; // Skip if no data to display

		// Define margins around the chart area
		const margin = { top: 20, right: 40, bottom: 20, left: 40 };
		const innerWidth = width - margin.left - margin.right; // Available width for chart
		const innerHeight = height - margin.top - margin.bottom; // Available height for chart

		// Clear any previous chart content
		d3.select(svg).selectAll('*').remove();

		// Clear brushes object but preserve current filter state
		const currentFilters = { ...brushFilters };
		brushes = {};

		// Create main SVG group with proper positioning
		const svgElement = d3
			.select(svg)
			.attr('width', width)
			.attr('height', height)
			.append('g')
			.attr('transform', `translate(${margin.left}, ${margin.top})`); // Offset by margins

		// Create scales for mapping data values to pixel coordinates
		const newScales = createScales(innerHeight, margin);

		// Create scale for positioning dimensions horizontally
		const xScale = d3
			.scalePoint()
			.domain(dimensions.map((d) => d.name)) // All dimension names
			.range([0, innerWidth]) // Spread across available width
			.padding(0.1); // Small padding between axes

		// Create line generator for drawing data paths
		const line = createLineGenerator(newScales, xScale);

		// Create color scale for axis identification
		const axisColorScale = d3
			.scaleOrdinal<string, string>()
			.domain(dimensions.map((d) => d.name))
			.range(COLOR_PALETTE); // Use predefined color palette

		// Draw preferred ranges first (behind everything else)
		drawPreferredRanges(svgElement, newScales, xScale);

		// Draw axes and labels
		dimensions.forEach((dim) => {
			const x = xScale(dim.name)!; // Get x position for this dimension
			const axisColor = axisColorScale(dim.name); // Get color for this dimension

			// Draw axis line with tick marks and labels
			svgElement
				.append('g')
				.attr('class', `axis axis-${dim.name}`)
				.attr('transform', `translate(${x}, 0)`) // Position at correct x coordinate
				.call(d3.axisLeft(newScales[dim.name]).ticks(5)); // Left-aligned axis with 5 ticks

			// Draw axis labels and colored identification squares
			if (options.showAxisLabels) {
				// Colored square for visual identification of each axis
				svgElement
					.append('rect')
					.attr('class', 'axis-color-square')
					.attr('x', x - 20) // Position to the left of the axis
					.attr('y', -18) // Position above the chart area
					.attr('width', 10)
					.attr('height', 10)
					.attr('fill', axisColor) // Use dimension's assigned color
					.attr('stroke', '#333') // Dark border
					.attr('stroke-width', 1)
					.attr('rx', 2) // Rounded corners
					.attr('ry', 2);

				// Axis name with direction indicator
				svgElement
					.append('text')
					.attr('class', 'axis-label')
					.attr('x', x - 5) // Position to the right of the colored square
					.attr('y', -8) // Position just above the chart area
					.attr('text-anchor', 'start') // Left-align text
					.style('font-size', '12px')
					.style('font-weight', 'bold')
					.style('fill', '#333')
					.text(dim.direction ? `${dim.name} (${dim.direction})` : dim.name); // Include direction if specified
			}
		});

		// Draw main data lines
		const lines = svgElement
			.append('g')
			.attr('class', 'data-lines')
			.selectAll('path')
			.data(data) // Bind data array
			.join('path') // Create path element for each data point
			.attr('d', (d, i) => {
				// Convert data point to line coordinates
				const lineData: [string, number][] = dimensions
					.map((dim) => [dim.name, d[dim.name]])
					.filter(([, value]) => value !== undefined && value !== null);
				return line(lineData); // Generate SVG path string
			})
			.attr('fill', 'none') // Lines have no fill, only stroke
			.attr('class', (d, i) => `line line-${i}`) // Unique class for each line
			.style('cursor', 'pointer'); // Show pointer cursor to indicate clickability

		// Set up brushing for each axis (must be done before line updates)
		dimensions.forEach((dim) => {
			const x = xScale(dim.name)!;
			setupAxisBrushing(svgElement, dim.name, x, innerHeight, lines);
		});

		// Apply initial line styling based on current state
		updateLineVisibility(lines);

		// Add hover effects if enabled
		if (options.highlightOnHover) {
			lines
				.on('mouseover', function (event, d) {
					if (!passesFilters(d)) return; // Only highlight visible lines

					// Temporarily increase stroke width on hover
					d3.select(this).attr('stroke-width', options.strokeWidth + 2);
				})
				.on('mouseout', function (event, d) {
					const index = data.indexOf(d);

					// Restore original stroke width
					d3.select(this).attr(
						'stroke-width',
						index === selectedIndex ? options.strokeWidth + 1 : options.strokeWidth
					);
				});
		}

		// Add click handler for line selection
		lines.on('click', function (event, d) {
			if (!passesFilters(d)) return; // Only allow clicking on visible lines

			const index = data.indexOf(d);
			handleLineClick(index, d); // Handle selection logic

			updateLineVisibility(lines); // Update visual state
		});

		// Draw reference visualizations (on top of data lines)
		drawReferencePoint(svgElement, newScales, xScale, line);
		drawReferenceSolutions(svgElement, newScales, xScale, line);

		// Add filter status information at the bottom
		const activeFilters = Object.keys(brushFilters).length;
		if (activeFilters > 0) {
			const visibleLines = data.filter(passesFilters).length;
			svgElement
				.append('text')
				.attr('class', 'filter-info')
				.attr('x', 10)
				.attr('y', innerHeight + 25) // Position below the chart
				.style('font-size', '11px')
				.style('fill', '#666')
				.text(
					`Showing ${visibleLines} of ${data.length} solutions (${activeFilters} filter${activeFilters > 1 ? 's' : ''} active)`
				);
		}
	}

	// --- Lifecycle Management ---

	/**
	 * Component initialization
	 * Sets up responsive behavior and draws initial chart
	 */
	onMount(() => {
		// Set up responsive behavior using ResizeObserver
		resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const rect = entry.contentRect;
				width = rect.width; // Update width when container resizes
				height = rect.height; // Update height when container resizes
				drawChart(); // Redraw chart with new dimensions
			}
		});
		resizeObserver.observe(container); // Start observing the container
		drawChart(); // Draw initial chart
	});

	/**
	 * Component cleanup
	 * Disconnect observers to prevent memory leaks
	 */
	onDestroy(() => {
		resizeObserver?.disconnect();
	});

	// --- Reactive Updates ---
	// Redraw chart whenever any of these values change
	$: data,
		dimensions,
		options,
		referenceData,
		selectedIndex,
		brushFilters,
		width,
		height,
		drawChart();
</script>

<!--
    Responsive container for the parallel coordinates plot.
    Uses aspect ratio to maintain consistent proportions.
-->
<div bind:this={container} style="height: 100%;width: 100%;">
	<svg bind:this={svg} style="width: 100%; height: 100%;" />
</div>

<style>
	/* Axis styling */
	:global(.axis) {
		font-size: 11px;
	}

	:global(.axis path),
	:global(.axis line) {
		fill: none;
		stroke: #000;
		shape-rendering: crispEdges;
	}

	/* Data line animations */
	:global(.line) {
		transition:
			stroke-width 0.2s,
			opacity 0.2s;
	}

	/* Text styling */
	:global(.axis-label) {
		fill: #333;
	}

	:global(.direction-label) {
		fill: #666;
	}

	:global(.selection-title) {
		fill: #333;
	}

	:global(.filter-info) {
		fill: #666;
	}

	/* Colored squares for axis identification */
	:global(.axis-color-square) {
		stroke: #333;
		stroke-width: 1;
		rx: 2;
		ry: 2;
	}

	/* Reference data styling - prevent interaction */
	:global(.preferred-ranges rect) {
		pointer-events: none;
	}

	:global(.reference-point path) {
		pointer-events: none;
	}

	:global(.reference-point circle) {
		pointer-events: none;
	}

	:global(.preferred-solutions path),
	:global(.non-preferred-solutions path) {
		pointer-events: none;
	}

	/* Brush styling */
	:global(.brush .selection) {
		fill: #4a90e2;
		opacity: 0.1;
	}

	:global(.brush .handle) {
		fill: #4a90e2;
		stroke: #4a90e2;
		stroke-width: 2;
	}

	/* Brush highlight rectangles */
	:global(.brush-highlight) {
		fill: #4a90e2;
		opacity: 0.2;
		stroke: #4a90e2;
		stroke-width: 2;
		stroke-dasharray: 3, 3;
		pointer-events: none;
		transition: opacity 0.2s ease;
	}

	:global(.brush-highlight:hover) {
		opacity: 0.3;
	}
</style>
