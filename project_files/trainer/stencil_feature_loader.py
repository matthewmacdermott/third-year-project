"""Stencil feature extraction using regex-based coordinate parsing.

Extracts raw stencil coordinate points from OPS source code for use in RL.
Instead of computing derived metrics (symmetry, aspect ratio, etc.), this
approach directly encodes stencil access patterns as x,y,z coordinates,
allowing the neural network to learn which patterns matter for HLS optimization.

Example:
    int s2d_5pt[] = {0,0, 1,0, -1,0, 0,1, 0,-1};

Extracts: [(0,0), (1,0), (-1,0), (0,1), (0,-1)] → encodes as raw coordinates
"""

import json
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional


class StencilFeatureLoader:
    """Extract and encode raw stencil coordinates for RL state representation.
    
    Uses regex parsing to find stencil coordinate arrays and encodes
    coordinates directly rather than computing derived features, allowing
    neural networks to learn optimal patterns from raw spatial data.
    """
    
    def __init__(self):
        """Initialize the feature loader."""
        pass
    
    def extract_from_source(self, source_file: Path) -> Dict:
        """
        Extract stencil features using simple regex pattern matching.
        
        Args:
            source_file: Path to .cpp file
            
        Returns:
            Dictionary with stencil features
        """
        with open(source_file, 'r') as f:
            content = f.read()
        
        # Extract dimension from OPS_2D/OPS_3D macro.
        if 'OPS_3D' in content:
            ndim = 3
        elif 'OPS_2D' in content:
            ndim = 2
        else:
            return {
                "file_info": {
                    "source_file": str(source_file),
                    "app_name": source_file.parent.parent.name,
                    "platform": source_file.parent.name,
                    "program_dimensions": None
                },
                "stencils": [],
                "program_dimensions": None,
                "total_stencils": 0
            }
        
        # Parse stencil arrays using regex
        stencils = []
        
        # First, find all stencil array definitions to get the coordinate points
        array_coords = {}  # Maps array name to list of points
        array_pattern = r'int\s+([sS]\w*)\[\]\s*=\s*\{([^}]+)\}'
        
        for match in re.finditer(array_pattern, content):
            array_name = match.group(1)
            coords_str = match.group(2)
            try:
                coords = [int(x.strip()) for x in coords_str.split(',')]
                if len(coords) % ndim == 0:
                    points_list = [coords[i:i+ndim] for i in range(0, len(coords), ndim)]
                    array_coords[array_name] = points_list
            except ValueError:
                continue  # Skip non-numeric arrays
        
        # Main path: build stencils directly from parsed coordinate arrays.
        for array_name, coord_points in array_coords.items():
            points_dicts = []
            max_radius = 0
            
            for pt in coord_points:
                point_dict = {"x": pt[0], "y": pt[1] if len(pt) > 1 else 0, "z": pt[2] if len(pt) > 2 else 0}
                points_dicts.append(point_dict)
                radius = max(abs(pt[0]), abs(pt[1]) if len(pt) > 1 else 0, abs(pt[2]) if len(pt) > 2 else 0)
                max_radius = max(max_radius, radius)
            
            stencil_info = {
                "name": array_name,
                "dimensions": ndim,
                "num_points": len(coord_points),
                "max_radius": int(max_radius),
                "points": points_dicts
            }
            stencils.append(stencil_info)
        
        if not stencils:
            raise ValueError(
                f"No stencil coordinate arrays parsed in {source_file}. "
                f"Detected OPS_{ndim}D source. "
                "Expected int s*[] or S*[] array declarations with numeric coordinates."
            )
        
        return {
            "file_info": {
                "source_file": str(source_file),
                "app_name": source_file.parent.parent.name,
                "platform": source_file.parent.name,
                "program_dimensions": ndim
            },
            "stencils": stencils,
            "program_dimensions": ndim,
            "total_stencils": len(stencils)
        }
    
    def load_from_json(self, json_path: str) -> Dict:
        """
        Load stencil features from extracted JSON file.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('stencil_features', {})
    
    def load_from_project(self, project_dir: Path) -> Optional[Dict]:
        """
        Load stencil features for a given project directory.
        First tries JSON files, then falls back to regex parsing.
        
        Args:
            project_dir: Path to the HLS project directory
            
        Returns:
            Stencil features dict or None if not found
        """
        # Try JSON files first (faster and more reliable)
        project_name = project_dir.name  # e.g., "u280_project"
        app_name = project_dir.parent.name  # e.g., "jacobian2d"
        
        # Check in extracted_features directory
        features_dir = Path("extracted_features")
        if not features_dir.is_absolute():
            features_dir = Path.cwd() / features_dir
        
        # Look for matching feature file with app name
        pattern = f"{app_name}_{project_name}_features.json"
        feature_files = list(features_dir.glob(pattern))
        
        if feature_files:
            return self.load_from_json(str(feature_files[0]))
        
        # Fallback to regex parsing with source files  
        cpp_files = list(project_dir.glob("*.cpp"))
        if cpp_files:
            stencil_data = self.extract_from_source(cpp_files[0])
            if stencil_data:
                return stencil_data
        
        # No stencil features available - this will cause the system to fail properly
        raise FileNotFoundError(f"No stencil features found for {app_name}_{project_name}. Create the JSON file or fix source parsing.")
    
    def extract_features_vector(self, stencil_data: Dict, max_points: int = 27) -> np.ndarray:
        """
        Convert parsed stencil metadata into a fixed-width state vector.
        
        Args:
            stencil_data: Dictionary containing stencil features from JSON
            max_points: Maximum number of stencil points to encode before truncation.
                The default of 27 matches the state layout used by the agent.
            
        Returns:
            Vector containing:
            - [0]: Program dimensions (2 or 3)
            - [1]: Stencil dimensions (2 or 3)
            - [2]: Number of points in largest stencil
            - [3-5]: Point 0 coordinates (x, y, z)
            - [6-8]: Point 1 coordinates (x, y, z)
            - ... up to max_points
            
            Total size: 3 + (max_points * 3) = 84 when max_points=27.
        """
        if not stencil_data or 'stencils' not in stencil_data:
            raise ValueError("No stencil data provided - cannot extract features")
        
        stencils = stencil_data['stencils']
        file_info = stencil_data.get('file_info', {})
        
        if not stencils:
            app_name = file_info.get('app_name', 'unknown')
            raise ValueError(f"No stencils found in stencil data for {app_name} - stencil extraction failed")
        
        # Initialize vector: 3 metadata + (max_points * 3 coords)
        vector_size = 3 + (max_points * 3)
        feature_vector = np.zeros(vector_size, dtype=np.float32)
        
        # Get program and stencil dimensions
        program_dims = file_info.get('program_dimensions', 2)
        
        # Use the largest stencil so the state reflects the most complex access
        # pattern available for this application.
        largest = max(stencils, key=lambda s: s.get('num_points', 0))
        stencil_dims = largest.get('dimensions', program_dims)
        num_points = min(largest.get('num_points', 1), max_points)
        
        # Metadata — normalised to [0, 1] so they sit on the same scale as the
        # config features (which are also in [0, 1]).
        feature_vector[0] = program_dims / 3.0       # 2-D→0.667, 3-D→1.0
        feature_vector[1] = stencil_dims / 3.0       # same range
        feature_vector[2] = num_points / max_points  # [0, 1]
        
        # Stencil coordinates are scaled into a compact numeric range so the
        # network can consume them alongside the normalised configuration data.
        # A fixed divisor keeps common offsets small while still preserving sign
        # and relative distance.
        _COORD_SCALE = 5.0

        # Encode up to max_points coordinates and leave the remaining slots at 0.
        points = largest.get('points', [])[:max_points]
        for i, point in enumerate(points):
            offset = 3 + (i * 3)
            feature_vector[offset]     = point.get('x', 0) / _COORD_SCALE
            feature_vector[offset + 1] = point.get('y', 0) / _COORD_SCALE
            feature_vector[offset + 2] = point.get('z', 0) / _COORD_SCALE
        
        return feature_vector
    