#!/usr/bin/env python3
"""
Convert AIND multiplane ophys data format to suite2p format.

This script converts processed AIND calcium imaging data into suite2p-compatible
format, enabling visualization and analysis in the suite2p GUI.

Author: Generated for AIND team
Date: December 12, 2025
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import h5py
from tqdm import tqdm


class AINDToSuite2pConverter:
    """Converter for AIND multiplane ophys data to suite2p format."""
    
    def __init__(self, input_path: Path, output_path: Path, dataset_name: str,
                 plane_pattern: str = r"VISp_\d+", overwrite: bool = False,
                 validate: bool = True):
        """
        Initialize converter.
        
        Parameters
        ----------
        input_path : Path
            Path to AIND dataset root (contains VISp_X folders)
        output_path : Path
            Path for suite2p output root
        dataset_name : str
            Name of dataset for output folder structure
        plane_pattern : str
            Regex pattern to find plane folders
        overwrite : bool
            Whether to overwrite existing output
        validate : bool
            Whether to run validation checks
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        self.plane_pattern = re.compile(plane_pattern)
        self.overwrite = overwrite
        self.validate = validate
        
        # Output directory structure
        self.dataset_output_path = self.output_path / self.dataset_name
        
    def find_plane_folders(self) -> List[Tuple[str, Path]]:
        """
        Find all plane folders in input directory.
        
        Returns
        -------
        List[Tuple[str, Path]]
            List of (plane_name, plane_path) tuples, sorted by plane index
        """
        plane_folders = []
        
        for folder in self.input_path.iterdir():
            if folder.is_dir() and self.plane_pattern.match(folder.name):
                plane_folders.append((folder.name, folder))
        
        # Sort by extracting numeric index
        def extract_index(name_path):
            match = re.search(r'\d+', name_path[0])
            return int(match.group()) if match else 0
        
        plane_folders.sort(key=extract_index)
        
        print(f"Found {len(plane_folders)} plane folders: {[name for name, _ in plane_folders]}")
        return plane_folders
    
    def load_processing_metadata(self, plane_path: Path) -> Dict:
        """
        Load processing.json metadata for a plane.
        
        Parameters
        ----------
        plane_path : Path
            Path to plane folder
            
        Returns
        -------
        Dict
            Processing metadata including frame rate
        """
        # Try multiple possible locations for processing.json
        possible_paths = [
            plane_path / "processing.json",
            self.input_path / "processing.json",
        ]
        
        for json_path in possible_paths:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract frame rate
                try:
                    for process in metadata['processing_pipeline']['data_processes']:
                        if 'movie_frame_rate_hz' in process.get('parameters', {}):
                            return {
                                'fs': process['parameters']['movie_frame_rate_hz'],
                                'metadata': metadata
                            }
                except (KeyError, TypeError):
                    pass
        
        print(f"Warning: Could not find frame rate in processing.json, using default 10.0 Hz")
        return {'fs': 10.0, 'metadata': None}
    
    def unpack_sparse_coords(self, coords: np.ndarray, data: np.ndarray, 
                            n_rois: int) -> List[Dict[str, np.ndarray]]:
        """
        Unpack sparse coordinate format into per-ROI arrays.
        
        Parameters
        ----------
        coords : np.ndarray
            Shape (3, N) where coords[0]=roi_id, coords[1]=y, coords[2]=x
        data : np.ndarray
            Shape (N,) pixel weights
        n_rois : int
            Number of ROIs
            
        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of dicts with 'ypix', 'xpix', 'lam' for each ROI
        """
        roi_pixels = []
        
        for roi_id in range(n_rois):
            # Find pixels belonging to this ROI
            mask = coords[0] == roi_id
            
            roi_data = {
                'ypix': coords[1][mask].astype(np.int32),
                'xpix': coords[2][mask].astype(np.int32),
                'lam': data[mask].astype(np.float32)
            }
            
            roi_pixels.append(roi_data)
        
        return roi_pixels
    
    def unpack_sparse_bool(self, coords: np.ndarray, bool_data: np.ndarray,
                          n_rois: int) -> List[np.ndarray]:
        """
        Unpack sparse boolean arrays (overlap, soma_crop).
        
        Parameters
        ----------
        coords : np.ndarray
            Shape (3, N) where coords[0]=roi_id
        bool_data : np.ndarray
            Shape (N,) boolean flags
        n_rois : int
            Number of ROIs
            
        Returns
        -------
        List[np.ndarray]
            List of boolean arrays for each ROI
        """
        roi_bool = []
        
        for roi_id in range(n_rois):
            mask = coords[0] == roi_id
            roi_bool.append(bool_data[mask])
        
        return roi_bool
    
    def unpack_neuropil_coords(self, neuropil_coords: np.ndarray, n_rois: int, 
                               Lx: int) -> List[np.ndarray]:
        """
        Unpack neuropil coordinates and convert to linearized indices.
        
        Parameters
        ----------
        neuropil_coords : np.ndarray
            Shape (3, N) where coords[0]=roi_id, coords[1]=y, coords[2]=x
        n_rois : int
            Number of ROIs
        Lx : int
            Image width (needed to linearize coordinates)
            
        Returns
        -------
        List[np.ndarray]
            List of neuropil masks (linearized pixel indices) for each ROI
        """
        neuropil_masks = []
        
        for roi_id in range(n_rois):
            # Find neuropil pixels belonging to this ROI
            mask = neuropil_coords[0] == roi_id
            y_coords = neuropil_coords[1][mask].astype(np.int32)
            x_coords = neuropil_coords[2][mask].astype(np.int32)
            
            # Convert to linearized indices (same as np.ravel_multi_index)
            neuropil_indices = y_coords * Lx + x_coords
            
            neuropil_masks.append(neuropil_indices.astype(np.int64))
        
        return neuropil_masks
    
    def load_extraction_data(self, plane_path: Path) -> Dict:
        """
        Load extraction H5 file data.
        
        Parameters
        ----------
        plane_path : Path
            Path to plane folder
            
        Returns
        -------
        Dict
            Dictionary containing all extraction data
        """
        plane_name = plane_path.name
        h5_path = plane_path / "extraction" / f"{plane_name}_extraction.h5"
        
        if not h5_path.exists():
            raise FileNotFoundError(f"Extraction file not found: {h5_path}")
        
        print(f"  Loading extraction data from {h5_path.name}")
        
        with h5py.File(h5_path, 'r') as f:
            # Get dimensions
            n_rois = f['traces/roi'].shape[0]
            n_frames = f['traces/roi'].shape[1]
            
            # Load traces
            data = {
                'F': f['traces/corrected'][:].astype(np.float32),
                'Fneu': f['traces/neuropil'][:].astype(np.float32),
                'iscell': f['iscell'][:].astype(np.float32),
                'n_rois': n_rois,
                'n_frames': n_frames,
            }
            
            # Load ROI properties
            coords = f['rois/coords'][:]
            data_weights = f['rois/data'][:]
            
            # Unpack sparse coordinates
            roi_pixels = self.unpack_sparse_coords(coords, data_weights, n_rois)
            
            # Unpack sparse boolean arrays
            overlap = self.unpack_sparse_bool(coords, f['rois/overlap'][:], n_rois)
            soma_crop = self.unpack_sparse_bool(coords, f['rois/soma_crop'][:], n_rois)
            
            # Load other ROI properties
            data['roi_stats'] = {
                'med': f['rois/med'][:].astype(np.int16),
                'npix': f['rois/npix'][:].astype(np.int32),
                'npix_norm': f['rois/npix_norm'][:].astype(np.float32),
                'npix_soma': f['rois/npix_soma'][:].astype(np.int32),
                'radius': f['rois/radius'][:].astype(np.float32),
                'aspect_ratio': f['rois/aspect_ratio'][:].astype(np.float32),
                'compact': f['rois/compact'][:].astype(np.float32),
                'solidity': f['rois/solidity'][:].astype(np.float32),
                'footprint': f['rois/footprint'][:].astype(np.int16),
                'skew': f['traces/skew'][:].astype(np.float32),
                'std': f['traces/std'][:].astype(np.float32),
                'mrs': f['rois/mrs'][:].astype(np.float32) if 'rois/mrs' in f else None,
                'mrs0': f['rois/mrs0'][:].astype(np.float32) if 'rois/mrs0' in f else None,
                'neuropil_rcoef': f['traces/neuropil_rcoef'][:].astype(np.float32) if 'traces/neuropil_rcoef' in f else None,
                'raw_neuropil_rcoef_mutualinfo': f['traces/raw_neuropil_rcoef_mutualinfo'][:].astype(np.float32) if 'traces/raw_neuropil_rcoef_mutualinfo' in f else None,
            }
            
            # Add unpacked pixel data
            data['roi_pixels'] = roi_pixels
            data['overlap'] = overlap
            data['soma_crop'] = soma_crop
            
            # Load images
            data['meanImg'] = f['meanImg'][:].astype(np.float32)
            data['maxImg'] = f['maxImg'][:].astype(np.float32)
            
            # Get image dimensions
            data['Ly'], data['Lx'] = data['meanImg'].shape
            
            # Unpack neuropil masks if available (needs Lx)
            if 'rois/neuropil_coords' in f:
                neuropil_coords = f['rois/neuropil_coords'][:]
                data['neuropil_pixels'] = self.unpack_neuropil_coords(neuropil_coords, n_rois, data['Lx'])
            else:
                data['neuropil_pixels'] = None
            
        return data
    
    def load_classification_data(self, plane_path: Path) -> Optional[Dict]:
        """
        Load classification H5 file data.
        
        Parameters
        ----------
        plane_path : Path
            Path to plane folder
            
        Returns
        -------
        Optional[Dict]
            Dictionary containing classification data, or None if not found
        """
        plane_name = plane_path.name
        h5_path = plane_path / "classification" / f"{plane_name}_classification.h5"
        
        if not h5_path.exists():
            print(f"  Warning: Classification file not found: {h5_path.name}")
            return None
        
        print(f"  Loading classification data from {h5_path.name}")
        
        with h5py.File(h5_path, 'r') as f:
            data = {
                'soma_predictions': f['soma/predictions'][:].astype(np.int64),
                'soma_probabilities': f['soma/probabilities'][:].astype(np.float32),
            }
        
        return data
    
    def load_motion_correction_data(self, plane_path: Path) -> Dict:
        """
        Load motion correction H5 file data.
        
        Parameters
        ----------
        plane_path : Path
            Path to plane folder
            
        Returns
        -------
        Dict
            Dictionary containing motion correction data
        """
        plane_name = plane_path.name
        h5_path = plane_path / "motion_correction" / f"{plane_name}_registered.h5"
        
        if not h5_path.exists():
            print(f"  Warning: Motion correction file not found: {h5_path.name}")
            return {}
        
        print(f"  Loading motion correction data from {h5_path.name}")
        
        with h5py.File(h5_path, 'r') as f:
            data = {
                'refImg': f['ref_image'][:].astype(np.float32),
            }
            
            # Load registration metrics if available
            if 'reg_metrics/regDX' in f:
                regDX = f['reg_metrics/regDX'][:]
                # regDX shape: (n_chunks, 3) where columns are [y_shift, x_shift, correlation]
                data['yoff'] = regDX[:, 0].astype(np.float32)
                data['xoff'] = regDX[:, 1].astype(np.float32)
                data['corrXY'] = regDX[:, 2].astype(np.float32)
            
            if 'reg_metrics/crispness' in f:
                data['crispness'] = f['reg_metrics/crispness'][:].astype(np.float64)
        
        return data
    
    def load_events_data(self, plane_path: Path) -> Dict:
        """
        Load events (spike deconvolution) H5 file data.
        
        Parameters
        ----------
        plane_path : Path
            Path to plane folder
            
        Returns
        -------
        Dict
            Dictionary containing events data
        """
        plane_name = plane_path.name
        h5_path = plane_path / "events" / f"{plane_name}_events_oasis.h5"
        
        if not h5_path.exists():
            raise FileNotFoundError(f"Events file not found: {h5_path}")
        
        print(f"  Loading events data from {h5_path.name}")
        
        with h5py.File(h5_path, 'r') as f:
            data = {
                'spks': f['events'][:].astype(np.float32),
                'tau_hat': f['tau_hat'][:].astype(np.float32) if 'tau_hat' in f else None,
            }
        
        return data
    
    def build_stat_array(self, extraction_data: Dict, plane_idx: int) -> np.ndarray:
        """
        Build stat.npy array from extraction data.
        
        Parameters
        ----------
        extraction_data : Dict
            Extraction data dictionary
        plane_idx : int
            Plane index for iplane field
            
        Returns
        -------
        np.ndarray
            Object array of stat dictionaries
        """
        n_rois = extraction_data['n_rois']
        roi_pixels = extraction_data['roi_pixels']
        roi_stats = extraction_data['roi_stats']
        overlap = extraction_data['overlap']
        soma_crop = extraction_data['soma_crop']
        
        stat = np.zeros(n_rois, dtype=object)
        
        for i in range(n_rois):
            stat_dict = {
                'ypix': roi_pixels[i]['ypix'],
                'xpix': roi_pixels[i]['xpix'],
                'lam': roi_pixels[i]['lam'],
                'med': roi_stats['med'][i],
                'npix': roi_stats['npix'][i],
                'npix_norm': roi_stats['npix_norm'][i],
                'npix_soma': roi_stats['npix_soma'][i],
                'radius': roi_stats['radius'][i],
                'aspect_ratio': roi_stats['aspect_ratio'][i],
                'compact': roi_stats['compact'][i],
                'solidity': roi_stats['solidity'][i],
                'footprint': roi_stats['footprint'][i],
                'skew': roi_stats['skew'][i],
                'std': roi_stats['std'][i],
                'overlap': overlap[i],
                'soma_crop': soma_crop[i],
                'iplane': plane_idx,
            }
            
            # Add optional fields if available
            if roi_stats['mrs'] is not None:
                stat_dict['mrs'] = roi_stats['mrs'][i]
            if roi_stats['mrs0'] is not None:
                stat_dict['mrs0'] = roi_stats['mrs0'][i]
            if roi_stats['neuropil_rcoef'] is not None:
                stat_dict['neuropil_rcoef'] = roi_stats['neuropil_rcoef'][i]
            if roi_stats['raw_neuropil_rcoef_mutualinfo'] is not None:
                stat_dict['raw_neuropil_rcoef_mutualinfo'] = roi_stats['raw_neuropil_rcoef_mutualinfo'][i]
            
            # Add neuropil mask if available
            if extraction_data.get('neuropil_pixels') is not None:
                stat_dict['neuropil_mask'] = extraction_data['neuropil_pixels'][i]
            
            stat[i] = stat_dict
        
        return stat
    
    def build_ops_dict(self, extraction_data: Dict, motion_data: Dict,
                      metadata: Dict, plane_name: str, plane_idx: int,
                      n_planes: int) -> Dict:
        """
        Build ops.npy dictionary.
        
        Parameters
        ----------
        extraction_data : Dict
            Extraction data
        motion_data : Dict
            Motion correction data
        metadata : Dict
            Processing metadata
        plane_name : str
            Name of plane folder
        plane_idx : int
            Plane index
        n_planes : int
            Total number of planes
            
        Returns
        -------
        Dict
            ops dictionary
        """
        ops = {
            # Basic dimensions
            'Ly': extraction_data['Ly'],
            'Lx': extraction_data['Lx'],
            'nframes': extraction_data['n_frames'],
            
            # Images
            'meanImg': extraction_data['meanImg'],
            'max_proj': extraction_data['maxImg'],
            
            # Valid ranges (use full image for now)
            'yrange': np.array([0, extraction_data['Ly']]),
            'xrange': np.array([0, extraction_data['Lx']]),
            
            # Plane info
            'nplanes': n_planes,
            'nchannels': 1,
            'iplane': plane_idx,
            
            # Sampling rate
            'fs': metadata['fs'],
            
            # Cell detection parameters (for GUI compatibility)
            'diameter': 12,  # Default diameter, can be adjusted
            'spatial_scale': 0,
            'threshold_scaling': 1.0,
            
            # Paths
            'data_path': [str(self.input_path)],
            'save_path': str(self.dataset_output_path / plane_name),
            
            # Timestamp
            'date_proc': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Source info
            'input_format': 'AIND',
            'plane_name': plane_name,
        }
        
        # Add motion correction data if available
        if 'refImg' in motion_data:
            ops['refImg'] = motion_data['refImg']
        
        if 'yoff' in motion_data:
            ops['yoff'] = motion_data['yoff']
            ops['xoff'] = motion_data['xoff']
            ops['corrXY'] = motion_data['corrXY']
        
        if 'crispness' in motion_data:
            ops['crispness'] = motion_data['crispness']
        
        # Add tau if available from events
        # Will be set by caller if events data is loaded
        
        return ops
    
    def build_iscell_alt(self, classification_data: Dict, n_rois: int) -> np.ndarray:
        """
        Build iscell_alt.npy from classification data.
        
        Parameters
        ----------
        classification_data : Dict
            Classification data
        n_rois : int
            Number of ROIs
            
        Returns
        -------
        np.ndarray
            Shape (n_rois, 2) with [prediction, probability]
        """
        iscell_alt = np.zeros((n_rois, 2), dtype=np.float32)
        iscell_alt[:, 0] = classification_data['soma_predictions'].astype(np.float32)
        # Use probability of soma class (index 1)
        iscell_alt[:, 1] = classification_data['soma_probabilities'][:, 1]
        return iscell_alt
    
    def convert_plane(self, plane_name: str, plane_path: Path, plane_idx: int,
                     n_planes: int) -> bool:
        """
        Convert a single plane from AIND to suite2p format.
        
        Parameters
        ----------
        plane_name : str
            Name of plane folder
        plane_path : Path
            Path to plane folder
        plane_idx : int
            Plane index
        n_planes : int
            Total number of planes
            
        Returns
        -------
        bool
            True if successful
        """
        print(f"\nConverting plane: {plane_name} (index {plane_idx})")
        
        try:
            # Create output directory
            output_dir = self.dataset_output_path / plane_name
            
            if output_dir.exists() and not self.overwrite:
                print(f"  Output directory exists and overwrite=False, skipping: {output_dir}")
                return False
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load processing metadata
            metadata = self.load_processing_metadata(plane_path)
            print(f"  Frame rate: {metadata['fs']} Hz")
            
            # Load all data
            extraction_data = self.load_extraction_data(plane_path)
            classification_data = self.load_classification_data(plane_path)
            motion_data = self.load_motion_correction_data(plane_path)
            events_data = self.load_events_data(plane_path)
            
            n_rois = extraction_data['n_rois']
            n_frames = extraction_data['n_frames']
            print(f"  Data dimensions: {n_rois} ROIs, {n_frames} frames")
            
            # Save basic arrays
            print(f"  Saving F.npy")
            np.save(output_dir / 'F.npy', extraction_data['F'])
            
            print(f"  Saving Fneu.npy")
            np.save(output_dir / 'Fneu.npy', extraction_data['Fneu'])
            
            print(f"  Saving spks.npy")
            np.save(output_dir / 'spks.npy', events_data['spks'])
            
            print(f"  Saving iscell.npy")
            np.save(output_dir / 'iscell.npy', extraction_data['iscell'])
            
            # Save alternative classification if available
            if classification_data is not None:
                print(f"  Saving iscell_alt.npy")
                iscell_alt = self.build_iscell_alt(classification_data, n_rois)
                np.save(output_dir / 'iscell_alt.npy', iscell_alt)
            
            # Build and save stat.npy
            print(f"  Building stat.npy")
            stat = self.build_stat_array(extraction_data, plane_idx)
            np.save(output_dir / 'stat.npy', stat, allow_pickle=True)
            
            # Build and save ops.npy
            print(f"  Building ops.npy")
            ops = self.build_ops_dict(extraction_data, motion_data, metadata,
                                     plane_name, plane_idx, n_planes)
            
            # Add mean tau if available
            if events_data['tau_hat'] is not None:
                ops['tau'] = float(np.mean(events_data['tau_hat']))
            else:
                ops['tau'] = 1.0
            
            np.save(output_dir / 'ops.npy', ops, allow_pickle=True)
            
            print(f"  ✓ Successfully converted {plane_name}")
            
            if self.validate:
                self.validate_plane_output(output_dir, n_rois, n_frames)
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error converting {plane_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_plane_output(self, output_dir: Path, expected_rois: int,
                             expected_frames: int):
        """
        Validate converted output files.
        
        Parameters
        ----------
        output_dir : Path
            Output directory to validate
        expected_rois : int
            Expected number of ROIs
        expected_frames : int
            Expected number of frames
        """
        print(f"  Validating output...")
        
        errors = []
        
        # Check all required files exist
        required_files = ['F.npy', 'Fneu.npy', 'spks.npy', 'stat.npy', 'ops.npy', 'iscell.npy']
        for fname in required_files:
            if not (output_dir / fname).exists():
                errors.append(f"Missing file: {fname}")
        
        if errors:
            print(f"  ✗ Validation errors:")
            for error in errors:
                print(f"    - {error}")
            return
        
        # Load and check dimensions
        try:
            F = np.load(output_dir / 'F.npy')
            Fneu = np.load(output_dir / 'Fneu.npy')
            spks = np.load(output_dir / 'spks.npy')
            stat = np.load(output_dir / 'stat.npy', allow_pickle=True)
            iscell = np.load(output_dir / 'iscell.npy')
            ops = np.load(output_dir / 'ops.npy', allow_pickle=True).item()
            
            # Check shapes
            if F.shape != (expected_rois, expected_frames):
                errors.append(f"F.npy shape mismatch: {F.shape} != ({expected_rois}, {expected_frames})")
            
            if Fneu.shape != (expected_rois, expected_frames):
                errors.append(f"Fneu.npy shape mismatch: {Fneu.shape} != ({expected_rois}, {expected_frames})")
            
            if spks.shape != (expected_rois, expected_frames):
                errors.append(f"spks.npy shape mismatch: {spks.shape} != ({expected_rois}, {expected_frames})")
            
            if len(stat) != expected_rois:
                errors.append(f"stat.npy length mismatch: {len(stat)} != {expected_rois}")
            
            if iscell.shape != (expected_rois, 2):
                errors.append(f"iscell.npy shape mismatch: {iscell.shape} != ({expected_rois}, 2)")
            
            # Check ops has required fields
            required_ops_fields = ['Ly', 'Lx', 'nframes', 'meanImg', 'fs']
            for field in required_ops_fields:
                if field not in ops:
                    errors.append(f"ops.npy missing field: {field}")
            
            if errors:
                print(f"  ✗ Validation errors:")
                for error in errors:
                    print(f"    - {error}")
            else:
                print(f"  ✓ Validation passed")
                
        except Exception as e:
            print(f"  ✗ Validation error: {e}")
    
    def convert_all(self) -> int:
        """
        Convert all planes in the dataset.
        
        Returns
        -------
        int
            Number of successfully converted planes
        """
        print(f"\n{'='*60}")
        print(f"AIND to Suite2p Conversion")
        print(f"{'='*60}")
        print(f"Input:  {self.input_path}")
        print(f"Output: {self.dataset_output_path}")
        print(f"{'='*60}\n")
        
        # Find all plane folders
        plane_folders = self.find_plane_folders()
        
        if not plane_folders:
            print("Error: No plane folders found!")
            return 0
        
        # Convert each plane
        n_success = 0
        n_planes = len(plane_folders)
        
        for plane_idx, (plane_name, plane_path) in enumerate(plane_folders):
            success = self.convert_plane(plane_name, plane_path, plane_idx, n_planes)
            if success:
                n_success += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Conversion Summary")
        print(f"{'='*60}")
        print(f"Total planes: {n_planes}")
        print(f"Successful:   {n_success}")
        print(f"Failed:       {n_planes - n_success}")
        print(f"{'='*60}\n")
        
        return n_success


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description='Convert AIND multiplane ophys data to suite2p format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_aind_to_suite2p.py \\
      --input /s3-cache/suite2p-dev/multiplane-ophys_827543_2025-12-11_14-26-35_processed_2025-12-12_13-39-05 \\
      --output /output/suite2p_converted \\
      --dataset-name multiplane-ophys_827543 \\
      --validate

Output structure:
  /output/suite2p_converted/
  └── multiplane-ophys_827543/
      ├── VISp_0/
      │   ├── F.npy
      │   ├── Fneu.npy
      │   ├── spks.npy
      │   ├── stat.npy
      │   ├── ops.npy
      │   ├── iscell.npy
      │   └── iscell_alt.npy
      ├── VISp_1/
      └── ...
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to AIND dataset root (contains VISp_X folders)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path for suite2p output root'
    )
    
    parser.add_argument(
        '--dataset-name', '-n',
        type=str,
        required=True,
        help='Name of dataset for output folder structure'
    )
    
    parser.add_argument(
        '--plane-pattern', '-p',
        type=str,
        default=r'VISp_\d+',
        help='Regex pattern to find plane folders (default: VISp_\\d+)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation checks'
    )
    
    args = parser.parse_args()
    
    # Create converter and run
    converter = AINDToSuite2pConverter(
        input_path=args.input,
        output_path=args.output,
        dataset_name=args.dataset_name,
        plane_pattern=args.plane_pattern,
        overwrite=args.overwrite,
        validate=not args.no_validate
    )
    
    n_success = converter.convert_all()
    
    # Exit with appropriate code
    if n_success == 0:
        exit(1)
    elif n_success < len(converter.find_plane_folders()):
        exit(2)  # Partial success
    else:
        exit(0)  # Full success


if __name__ == '__main__':
    main()
