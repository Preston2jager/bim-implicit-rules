import os
import re
import shutil
import random
import math
import subprocess
import yaml
import json
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from datetime import date
from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Optional, Any

import numpy as np
import trimesh
import point_cloud_utils as pcu
from trimesh import repair
from tqdm import tqdm
from loguru import logger

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.guid
import ifcopenshell.util.element
import ifcopenshell.util.unit


def _sample_obj_worker(job: Tuple[int, str, dict, str, int]) -> Tuple[int, int]:
    """
    Subprocess worker for _stage_sample_sdf: samples ONE OBJ and writes a compact
    float32 part file to <parts_dir>/<obj_idx>.npz. The heavy calls (pcu mesh
    sampling + signed-distance BVH queries) hold the GIL, so threads can't scale
    them -- one process per core does. Must stay module-level (picklable).
    """
    obj_idx, obj_path, params, parts_dir, seed = job
    np.random.seed((seed + obj_idx) % (2 ** 32))
    dp = DataPreprocessor(SimpleNamespace())  # helpers below don't touch config

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    mesh = dp._clean_mesh(mesh)
    if mesh is None:
        return obj_idx, 0
    if not mesh.is_watertight:
        logger.warning(f"Repairing non-watertight mesh: {obj_path}")
        repair.fill_holes(mesh)
        repair.fix_normals(mesh)
    assert mesh.is_watertight, f"CRITICAL: Mesh {obj_path} is NOT watertight."

    total_area = dp._compute_triangle_areas(mesh.vertices, mesh.faces).sum()
    p_total, sdf, _ = dp._sample_points_and_compute_sdf(
        mesh=mesh,
        total_area=total_area,
        scale=params["scale"],
        radii=params["radii"],
        nums=params["nums"],
        min_thickness=params["min_thickness"],
        density_per_m2=params["density_per_m2"],
        min_per_surface=params["min_per_surface"],
    )
    if p_total.size == 0:
        return obj_idx, 0
    np.savez(
        Path(parts_dir) / f"{obj_idx:06d}.npz",
        points=p_total.astype(np.float32),
        sdf=sdf.astype(np.float32),
    )
    return obj_idx, int(p_total.shape[0])


class DataPreprocessor:
    """
    Consolidated data preprocessing pipeline for IFC files.
    Integrates workspace initialization, data expansion, OBJ export (with hole filling),
    and watertight point cloud sampling.
    """

    # ===== Initialisation and Config Loading =====

    def __init__(self, config: SimpleNamespace):
        self.config = config
        self._export_date = date.today().strftime("%Y%m%d")

    @classmethod
    def config_load(cls, yaml_path: Path = Path("config.yaml")) -> "DataPreprocessor":
        """
        Loads the YAML configuration file and returns a DataPreprocessor instance.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        def dict_to_obj(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
            if isinstance(d, list):
                return [dict_to_obj(i) for i in d]
            return d
            
        config_obj = dict_to_obj(data)
        logger.success(f"Config file loaded from: {yaml_path}")
        return cls(config_obj)

    # ===== Public APIs =====

    @classmethod
    def execute(cls, config_path: Path, work_folder: Path) -> None:
        """
        Single-line pipeline call.
        Loads the configuration, instantiates the DataPreprocessor, runs all 3 stages,
        and releases the instance.
        """
        processor = cls.config_load(config_path)
        try:
            processor._run(work_folder)
        finally:
            # Explicitly release references to release resources
            del processor
            logger.info("DataPreprocessor instance has been released.")

    @classmethod
    def generate_graphs(cls, config_path: Path, work_folder: Path) -> None:
        """
        Separate entry point to generate spatial graphs from the IFC files.
        Detects if embedded IFCs exist in intermediate/embedded_ifcs/,
        otherwise falls back to intermediate/expanded_ifcs/.
        """
        processor = cls.config_load(config_path)
        try:
            processor._run_graph_generation(work_folder)
        finally:
            del processor
            logger.info("DataPreprocessor instance has been released.")

    # ===== Private APIs =====

    def _run(self, work_folder: Path) -> None:
        """
        Executes the three initial preprocessing stages sequentially.
        """
        work_folder = Path(work_folder)
        logger.info(f"=== Starting Preprocessing Pipeline in: {work_folder} ===")
        
        # Stage 1: Data expansion/augmentation
        self._stage_expand_data(work_folder)
        
        # Stage 2: IFC Export to OBJ
        self._stage_export_objs(work_folder)
        
        # Stage 3: Watertight point/SDF sampling
        self._stage_sample_sdf(work_folder)
        
        logger.success("=== Preprocessing Pipeline Completed ===")

    def _run_graph_generation(self, work_folder: Path) -> None:
        """
        Executes only the spatial graph export stage.
        """
        work_folder = Path(work_folder)
        logger.info(f"=== Starting Spatial Graph Generation in: {work_folder} ===")
        self._stage_export_graphs(work_folder)
        logger.success("=== Graph Generation Completed ===")


    # ===== Private APIs: Stage 1 (Data Expansion) =====

    def _stage_expand_data(self, work_folder: Path) -> None:
        """
        Cleans the target folder and copies/expands IFC files.
        """
        expanded_folder = work_folder / "intermediate" / "expanded_ifcs"
        raw_folder = work_folder / "input" / "input_ifcs"
        
        logger.info("Stage 1/3: Expanding IFC files...")
        self._clear_folder(expanded_folder)
        self._copy_to_folder(raw_folder, expanded_folder)
        
        if getattr(self.config.extraction, "data_expand", False):
            self._expand_ifc_files(work_folder)
        else:
            logger.warning("Data expansion is disabled in config, skipping expansion.")

    def _expand_ifc_files(self, work_folder: Path) -> None:
        expanded_folder = work_folder / "intermediate" / "expanded_ifcs"
        ifc_files = list(expanded_folder.glob("*.ifc"))
        original_files = [f for f in ifc_files if "_aug_" not in f.name]
        
        num_copies = getattr(self.config.extraction, "copies", 0)
        if num_copies <= 0:
            logger.info("No copies requested in config. Skipping expansion.")
            return

        logger.info(f"Augmenting {len(original_files)} files with {num_copies} copies each.")
        for ifc_file_path in original_files:
            self._augment_single_file(ifc_file_path, num_copies)

    def _augment_single_file(self, file_path: Path, num_copies: int) -> None:
        try:
            template_model = ifcopenshell.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open IFC file {file_path.name}: {e}")
            return

        base_name = file_path.stem
        spatial_elements = template_model.by_type("IfcBuilding")
        if not spatial_elements:
            spatial_elements = template_model.by_type("IfcSite")
        
        if not spatial_elements:
            logger.warning(f"No IfcBuilding or IfcSite found in {file_path.name}. Cannot apply building-level transform.")
            return

        target_id = spatial_elements[0].id()
        rotation_range = (0, 2 * math.pi)

        # Copy translation is specified in SCALED coordinate units in the config
        # (world_meters * extraction.scale). Convert to this IFC's placement units so
        # the shift actually moves the geometry (and therefore the pose embeddings and
        # the absolute door target). NOTE: the previous hard-coded (-5, 5) was in file
        # units = millimetres, i.e. ~5 mm ≈ 0, which is why copies were rotation-only.
        scale = float(getattr(self.config.extraction, "scale", 1.0)) or 1.0
        t_max_scaled = float(getattr(self.config.extraction, "translation_max_scaled", 0.0))
        t_max_world_m = t_max_scaled / scale  # max Euclidean translation in metres
        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(template_model)  # metres per file unit
        logger.info(
            f"Copy translation: max {t_max_scaled} scaled = {t_max_world_m:.3f} m "
            f"({t_max_world_m / unit_scale:.1f} file units); rotation: full 360°."
        )

        for i in range(num_copies):
            try:
                model = ifcopenshell.open(file_path)
                target = model.by_id(target_id)

                angle = random.uniform(*rotation_range)
                # Uniform sample within a disk of radius t_max_world_m (metres), so the
                # Euclidean displacement never exceeds the configured bound.
                r_m = t_max_world_m * math.sqrt(random.random())
                phi = random.uniform(0, 2 * math.pi)
                tx = (r_m * math.cos(phi)) / unit_scale
                ty = (r_m * math.sin(phi)) / unit_scale
                tz = 0.0

                self._apply_transform(model, target, angle, (tx, ty, tz))
                self._regenerate_guids(model)
                
                aug_filename = f"{base_name}_aug_{i+1}.ifc"
                aug_path = file_path.parent / aug_filename
                model.write(str(aug_path))
                logger.info(f"  + Generated augmented file: {aug_filename}")
            except Exception as e:
                logger.error(f"Failed to generate augmented copy {i+1} for {file_path.name}: {e}")

    def _apply_transform(self, model, element, angle: float, translation: tuple) -> None:
        if not element.ObjectPlacement:
            logger.warning(f"Element {element.GlobalId} has no ObjectPlacement. Skipping transform.")
            return
            
        placement = element.ObjectPlacement
        if not placement.is_a("IfcLocalPlacement"):
            logger.warning(f"Unsupported placement type: {placement.is_a()}. Expected IfcLocalPlacement.")
            return

        axis2 = placement.RelativePlacement
        if not axis2 or not axis2.is_a("IfcAxis2Placement3D"):
            logger.warning(f"RelativePlacement is not IfcAxis2Placement3D. Cannot apply 3D transform.")
            return

        # Assign a DEDICATED cartesian point rather than mutating axis2.Location in place:
        # the origin point (0,0,0) is shared by ~all placements in the file, so mutating it
        # would shift every element that references it (compounding the intended translation).
        axis2.Location = model.create_entity("IfcCartesianPoint", Coordinates=translation)
        axis2.Axis = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        axis2.RefDirection = model.create_entity("IfcDirection", DirectionRatios=(math.cos(angle), math.sin(angle), 0.0))

    def _regenerate_guids(self, model) -> None:
        for entity in model.by_type("IfcRoot"):
            entity.GlobalId = ifcopenshell.guid.new()

    # ===== Private APIs: Stage 2 (IFC to OBJ Exporter) =====

    def _stage_export_objs(self, work_folder: Path) -> None:
        """
        Converts all IFC files in expanded folder to OBJ files.
        Implements the new logic (fills wall/slab openings, replaces doors/windows with AABBs).
        """
        logger.info("Stage 2/3: Exporting IFC elements to OBJ files...")
        objects_folder = work_folder / "intermediate" / "objects"
        expanded_folder = work_folder / "intermediate" / "expanded_ifcs"
        
        self._clear_folder(objects_folder)
        
        scale = getattr(self.config.extraction, "scale", 1.0)
        export_all_types = getattr(self.config.extraction, "Export_all_types", False)
        target_classes = getattr(self.config.extraction, "target_classes", None)
        
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, True)
        
        total_elements = 0
        file_index = 1
        # Precise record: OBJ file-index -> source IFC filename. This lets the
        # embedding write-back target each object's ORIGINAL IFC exactly, instead of
        # blindly trying every GUID against every IFC file.
        index2ifc: Dict[str, str] = {}

        for root, _, files in os.walk(expanded_folder):
            for filename in sorted(files):
                if filename.lower().endswith('.ifc'):
                    ifc_file_path = Path(root) / filename
                    index2ifc[str(file_index)] = ifc_file_path.name
                    elements_count = self._obj_export_single_file(
                        ifc_file_path=ifc_file_path,
                        output_dir=objects_folder,
                        index=file_index,
                        scale=scale,
                        export_all_types=export_all_types,
                        target_classes=target_classes,
                        settings=settings
                    )
                    total_elements += elements_count
                    file_index += 1

        index_map_path = work_folder / "intermediate" / "obj_index_to_ifc.json"
        with open(index_map_path, "w", encoding="utf-8") as f:
            json.dump(index2ifc, f, ensure_ascii=False, indent=2)
        logger.success(f"All IFC files from {expanded_folder} processed and exported to OBJ format.")
        logger.info(f"Scaled to {scale}. Total of {total_elements} elements converted to OBJ.")
        logger.info(f"OBJ index->IFC map written to: {index_map_path.name}")

    def _obj_export_single_file(
        self,
        ifc_file_path: Path,
        output_dir: Path,
        index: int,
        scale: float,
        export_all_types: bool,
        target_classes: Optional[List[str]],
        settings
    ) -> int:
        logger.info(f"Opening IFC file: {ifc_file_path}")
        try:
            ifc_file = ifcopenshell.open(str(ifc_file_path))
        except Exception as e:
            logger.error(f"Failed to open IFC file {ifc_file_path.name}: {e}")
            return 0

        classes_to_process = self._get_classes_to_process(ifc_file, export_all_types, target_classes)
        skip_classes = {"IfcOpeningElement"}
        classes_to_process = [c for c in classes_to_process if c not in skip_classes]
        special_classes = {"IfcWindow", "IfcDoor"}
        
        logger.info(f"Classes to export (excluding openings, doors/windows as AABB): {classes_to_process}")
        elements_exported = 0

        for class_name in classes_to_process:
            elements = ifc_file.by_type(class_name)
            if not elements:
                continue

            logger.info(f"Processing {class_name}: {len(elements)} objects.")
            for elem in elements:
                try:
                    elem_class = elem.is_a()
                    if elem_class in skip_classes:
                        continue

                    # Replace doors/windows with simplified bounding box (preserving world coordinates)
                    if elem_class in special_classes:
                        settings.set(settings.USE_WORLD_COORDS, False)
                        vertices, triangles = [], []
                        try:
                            shp = ifcopenshell.geom.create_shape(settings, elem)
                            verts_l = np.asarray(shp.geometry.verts, dtype=float).reshape(-1, 3)
                            T = np.asarray(shp.transformation.matrix, dtype=float).reshape(4, 4).T
                            lo, hi = verts_l.min(0), verts_l.max(0)
                            bbox_l = np.array([
                                [lo[0], lo[1], lo[2]],
                                [hi[0], lo[1], lo[2]],
                                [hi[0], hi[1], lo[2]],
                                [lo[0], hi[1], lo[2]],
                                [lo[0], lo[1], hi[2]],
                                [hi[0], lo[1], hi[2]],
                                [hi[0], hi[1], hi[2]],
                                [lo[0], hi[1], hi[2]],
                            ])
                            bbox_w = (T @ np.c_[bbox_l, np.ones(8)].T).T[:, :3]
                            vertices.extend(bbox_w)
                            faces_box = [
                                [0, 2, 1], [0, 3, 2],
                                [4, 5, 6], [4, 6, 7],
                                [0, 1, 5], [0, 5, 4],
                                [1, 2, 6], [1, 6, 5],
                                [2, 3, 7], [2, 7, 6],
                                [3, 0, 4], [3, 4, 7],
                            ]
                            if np.linalg.det(T[:3, :3]) < 0:
                                faces_box = [f[::-1] for f in faces_box]
                            triangles.extend(faces_box)
                        except Exception as e:
                            logger.warning(f"[{elem.GlobalId}] AABB export failed: {e}")
                            vertices, triangles = [], []

                        if vertices and triangles:
                            self._write_to_obj(
                                objs_dir=output_dir,
                                vertices=np.array(vertices),
                                faces=np.array(triangles),
                                uid=elem.GlobalId,
                                ifc_class=elem_class,
                                index=index,
                                scale=scale,
                                split=False,
                                groups=None
                            )
                            elements_exported += 1
                        continue

                    # Normal element exporting with hole filling
                    settings.set(settings.USE_WORLD_COORDS, True)
                    vertices, triangles = self._get_geometry(elem, settings)

                    # For walls/slabs with openings, merge opening geometry back to fill holes
                    rel_voids = []
                    try:
                        rel_voids = [r for r in ifc_file.by_type("IfcRelVoidsElement") if r.RelatingBuildingElement == elem]
                        for r in rel_voids:
                            opening = r.RelatedOpeningElement
                            op_verts, op_faces = self._get_geometry(opening, settings)
                            if len(op_verts) > 0 and len(op_faces) > 0:
                                vertices, triangles = self._merge_meshes(vertices, triangles, op_verts, op_faces)
                                logger.info(f"Merged opening {opening.GlobalId} geometry into {elem.GlobalId} ({elem_class}) to fill hole.")
                    except Exception as ex:
                        logger.warning(f"Failed to merge openings for element {elem.GlobalId}: {ex}")

                    if len(vertices) == 0 or len(triangles) == 0:
                        logger.warning(f"[{elem.GlobalId}] No geometry data, skipping")
                        continue

                    face_graph = self._build_face_graph(triangles)
                    segments = self._find_connected_components(triangles, face_graph)

                    # Disable splitting if we merged openings to keep host and patch in one file
                    split = len(segments) > 1 and not rel_voids
                    if split:
                        logger.info(f"Element {elem.GlobalId} has {len(segments)} separate parts; splitting.")
                        groups = segments
                    else:
                        groups = None

                    self._write_to_obj(
                        objs_dir=output_dir,
                        vertices=vertices,
                        faces=triangles,
                        uid=elem.GlobalId,
                        ifc_class=elem_class,
                        index=index,
                        scale=scale,
                        split=split,
                        groups=groups
                    )
                    elements_exported += 1
                except Exception as e:
                    logger.error(f"Error processing element {elem.GlobalId}: {e}")

        return elements_exported

    def _get_classes_to_process(self, ifc_file, export_all_types: bool, target_classes: Optional[List[str]] = None) -> List[str]:
        if export_all_types:
            detected_classes = set()
            base_classes = ['IfcBuildingElement', 'IfcSpatialElement', 'IfcElement']

            for base_class in base_classes:
                try:
                    elements = ifc_file.by_type(base_class)
                    for elem in elements:
                        if hasattr(elem, 'Representation') and elem.Representation is not None:
                            detected_classes.add(elem.is_a())
                except Exception:
                    pass

            if not detected_classes:
                common_types = [
                    'IfcWall', 'IfcSlab', 'IfcBeam', 'IfcColumn', 'IfcWindow', 'IfcDoor',
                    'IfcBuildingElementProxy', 'IfcSpace', 'IfcCovering', 'IfcFurnishingElement',
                    'IfcPlate', 'IfcMember', 'IfcFooting', 'IfcRoof', 'IfcStair', 'IfcRailing',
                    'IfcCurtainWall', 'IfcBuildingElementPart'
                ]
                for type_name in common_types:
                    if ifc_file.by_type(type_name):
                        detected_classes.add(type_name)

            return sorted(list(detected_classes))
        else:
            if not target_classes:
                target_classes = ['IfcWall', 'IfcSlab', 'IfcBeam', 'IfcColumn', 'IfcBuildingElementProxy']
            available_classes = []
            for class_name in target_classes:
                if ifc_file.by_type(class_name):
                    available_classes.append(class_name)
            return available_classes

    def _get_geometry(self, element, settings) -> Tuple[np.ndarray, np.ndarray]:
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            faces = np.array(shape.geometry.faces).reshape(-1, 3)
            return verts, faces
        except Exception as e:
            logger.error(f"Geometry extraction failed for element {element.GlobalId}: {e}")
            return np.array([]), np.array([])

    def _merge_meshes(
        self,
        verts1: np.ndarray, faces1: np.ndarray,
        verts2: np.ndarray, faces2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(verts1) == 0:
            return verts2, faces2
        if len(verts2) == 0:
            return verts1, faces1
            
        verts1 = np.asarray(verts1, dtype=float)
        faces1 = np.asarray(faces1, dtype=int)
        verts2 = np.asarray(verts2, dtype=float)
        faces2 = np.asarray(faces2, dtype=int)
        
        offset = len(verts1)
        merged_verts = np.vstack([verts1, verts2])
        merged_faces = np.vstack([faces1, faces2 + offset])
        
        return merged_verts, merged_faces

    def _build_face_graph(self, faces: np.ndarray) -> Dict[int, Set[int]]:
        edge_to_faces = defaultdict(set)
        face_graph = defaultdict(set)
        for i, face in enumerate(faces):
            edges = {
                (min(face[0], face[1]), max(face[0], face[1])),
                (min(face[1], face[2]), max(face[1], face[2])),
                (min(face[2], face[0]), max(face[2], face[0]))
            }
            for edge in edges:
                edge_to_faces[edge].add(i)
        for edge, face_set in edge_to_faces.items():
            face_list = list(face_set)
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    face_graph[face_list[i]].add(face_list[j])
                    face_graph[face_list[j]].add(face_list[i])
        return face_graph

    def _find_connected_components(self, faces: np.ndarray, face_graph: Dict[int, Set[int]]) -> List[Set[int]]:
        visited = set()
        segments = []

        def bfs(start_face):
            queue = deque([start_face])
            component = set()
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                queue.extend(face_graph[current] - visited)
            return component

        for i in range(len(faces)):
            if i not in visited:
                group = bfs(i)
                segments.append(group)
        return segments

    def _write_to_obj(
        self,
        objs_dir: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        uid: str,
        ifc_class: str,
        index: int,
        scale: float,
        split: bool = False,
        groups: Optional[List[Set[int]]] = None
    ) -> None:
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        vertices = vertices * scale

        objs_dir.mkdir(parents=True, exist_ok=True)

        if split and groups:
            for split_num, group in enumerate(groups, start=1):
                used_vertex_indices = set()
                for face_index in group:
                    used_vertex_indices.update(faces[face_index])
                used_vertex_indices = sorted(used_vertex_indices)
                index_mapping = {
                    old_idx: new_idx
                    for new_idx, old_idx in enumerate(used_vertex_indices, start=1)
                }
                
                # Compatible format: Date_Index_Class_GUID_splitNum.obj
                file_name = f"{self._export_date}_{index}_{ifc_class}_{uid}_{split_num}.obj"
                output_file = objs_dir / file_name
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("# Split OBJ File (segment export)\n")
                    for old_idx in used_vertex_indices:
                        vertex = vertices[old_idx]
                        vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                        f.write(vertex_str)
                    f.write(f"\no Object_{split_num}\n")
                    for face_index in group:
                        remapped = [index_mapping[v] for v in faces[face_index]]
                        f.write("f " + " ".join(str(idx) for idx in remapped) + "\n")
        else:
            # Compatible format: Date_Index_Class_GUID.obj
            file_name = f"{self._export_date}_{index}_{ifc_class}_{uid}.obj"
            output_file = objs_dir / file_name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# OBJ File\n")
                for vertex in vertices:
                    vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                    f.write(vertex_str)
                f.write("\no Object\n")
                for face in faces:
                    f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")

    # ===== Private APIs: Stage 3 (Sampling) =====

    def _stage_sample_sdf(self, work_folder: Path) -> None:
        """
        Samples points from meshes and calculates SDF datasets.

        Objects are sampled in parallel worker processes (the pcu sampling and
        signed-distance calls are GIL-bound per object), and results are streamed
        into flat float32 memmap arrays (samples_points/sdf/obj_idx.npy) instead of
        one giant pickled dict -- full-dataset runs no longer need the whole sample
        set in RAM at once.
        """
        logger.info("Stage 3/3: Sampling watertight point cloud and calculating SDF...")
        intermediate_folder = work_folder / "intermediate"
        self._clear_previous_sample_data(intermediate_folder)

        scale = self.config.extraction.scale
        params = {
            "scale": scale,
            "radii": [level.radius * scale for level in self.config.sampling.levels],
            "nums": [level.count for level in self.config.sampling.levels],
            "min_per_surface": int(self.config.sampling.min_per_surface),
            "density_per_m2": int(self.config.sampling.density_per_m2),
            "min_thickness": float(getattr(self.config.sampling, 'min_wall_thickness', 0.1)),
        }

        objects_dir = intermediate_folder / "objects"
        obj_files = list(objects_dir.glob("*.obj"))

        # Load the OBJ index -> source IFC map produced during export (for precise write-back).
        index2ifc = {}
        index_map_path = intermediate_folder / "obj_index_to_ifc.json"
        if index_map_path.exists():
            try:
                with open(index_map_path, "r", encoding="utf-8") as f:
                    index2ifc = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read obj_index_to_ifc.json: {e}")

        meta_dict = dict()
        idx_str2int_dict = dict()
        idx_int2str_dict = dict()
        idx_guid2int_dict = dict()
        class_name_to_idx = dict()
        class_idx_to_name = dict()

        if not obj_files:
            logger.warning(f"No OBJ files found in {objects_dir}. Sampling skipped.")
            return

        sorted_obj_files = sorted(obj_files, key=lambda p: p.name)

        # Identity/metadata pass (cheap, stays in the parent).
        valid_jobs = []
        for obj_idx, obj_path in enumerate(sorted_obj_files):
            obj_filename = obj_path.stem
            parts = obj_filename.split("_")
            if len(parts) < 4:
                continue
            file_idx = parts[1]
            ifc_class = parts[2]
            # IFC base64 GUIDs are always exactly 22 characters. Joining the remaining parts
            # handles both underscores inside the GUID and trailing split numbers (e.g., _1).
            guid = "_".join(parts[3:])[:22]
            if ifc_class not in class_name_to_idx:
                c_idx = len(class_name_to_idx)
                class_name_to_idx[ifc_class] = c_idx
                class_idx_to_name[c_idx] = ifc_class

            idx_str2int_dict[obj_filename] = obj_idx
            idx_int2str_dict[obj_idx] = obj_filename
            idx_guid2int_dict[guid] = obj_idx

            meta_dict[obj_idx] = {
                "guid": guid,
                "ifc_file_idx": file_idx,
                "source_ifc": index2ifc.get(str(file_idx)),
                "ifc_class": ifc_class,
                "class_idx": class_name_to_idx[ifc_class]
            }
            valid_jobs.append((obj_idx, str(obj_path)))

        # Preferred backend: the Rust sampler (sdf_sampler/ crate) -- one thread per
        # core, writes the flat arrays directly. Falls back to the Python
        # ProcessPool sampler when the crate/toolchain is unavailable.
        backend = str(getattr(self.config.sampling, "backend", "rust")).lower()
        stats = self._run_rust_sampler(valid_jobs, intermediate_folder) if backend == "rust" else None

        if stats is not None:
            total, pos, neg = int(stats["total"]), int(stats["pos"]), int(stats["neg"])
            zero = total - pos - neg
            logger.success("Sampling complete with strict architectural constraints. (Rust backend)")
            logger.info(f"SDF Statistics | Total: {total} | Pos: {pos/total:.1%} | Neg: {neg/total:.1%} | Zero: {zero/total:.1%}")
            logger.success(f"Flat sample arrays written ({total} points, float32, {stats['objects']} objects).")
        else:
            if backend == "rust":
                logger.warning("Falling back to the Python sampler.")
            # Parallel sampling: one part file per object.
            parts_dir = intermediate_folder / "sample_parts"
            shutil.rmtree(parts_dir, ignore_errors=True)
            parts_dir.mkdir(parents=True)
            seed = int(getattr(
                getattr(getattr(self.config, "training", None), "reproducibility", None),
                "seed", 42))

            max_workers = max(1, os.cpu_count() or 1)
            logger.info(f"Sampling {len(valid_jobs)} objects on {max_workers} worker processes...")
            counts: Dict[int, int] = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _sample_obj_worker, (obj_idx, path, params, str(parts_dir), seed)
                    ): obj_idx
                    for obj_idx, path in valid_jobs
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Sampling OBJ files"):
                    obj_idx = futures[fut]
                    try:
                        oid, n_points = fut.result()
                        if n_points > 0:
                            counts[oid] = n_points
                    except Exception as e:
                        logger.error(f"Failed processing {idx_int2str_dict.get(obj_idx, obj_idx)}: {e}")

            logger.success("Sampling complete with strict architectural constraints.")

            # Stream part files into flat memmap arrays; only one object in RAM at a time.
            if not counts:
                logger.warning("No data points were successfully sampled. Skipping statistics.")
            else:
                total = sum(counts.values())
                pts_mm = np.lib.format.open_memmap(
                    intermediate_folder / "samples_points.npy", mode="w+",
                    dtype=np.float32, shape=(total, 3))
                sdf_mm = np.lib.format.open_memmap(
                    intermediate_folder / "samples_sdf.npy", mode="w+",
                    dtype=np.float32, shape=(total,))
                idx_mm = np.lib.format.open_memmap(
                    intermediate_folder / "samples_obj_idx.npy", mode="w+",
                    dtype=np.int32, shape=(total,))
                cursor, pos, neg = 0, 0, 0
                bmin = np.full(3, np.inf)
                bmax = np.full(3, -np.inf)
                for oid in sorted(counts):
                    with np.load(parts_dir / f"{oid:06d}.npz") as part:
                        p, s = part["points"], part["sdf"]
                    n = p.shape[0]
                    pts_mm[cursor:cursor + n] = p
                    sdf_mm[cursor:cursor + n] = s
                    idx_mm[cursor:cursor + n] = oid
                    pos += int(np.sum(s > 1e-6))
                    neg += int(np.sum(s < -1e-6))
                    bmin = np.minimum(bmin, p.min(axis=0))
                    bmax = np.maximum(bmax, p.max(axis=0))
                    cursor += n
                pts_mm.flush(); sdf_mm.flush(); idx_mm.flush()
                del pts_mm, sdf_mm, idx_mm
                zero = total - pos - neg
                with open(intermediate_folder / "samples_stats.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "total": total, "pos": pos, "neg": neg, "zero": zero,
                        "bbox_min": bmin.tolist(), "bbox_max": bmax.tolist(),
                        "objects": len(counts),
                        "counts": [[int(k), int(v)] for k, v in sorted(counts.items())],
                    }, f)
                logger.info(f"SDF Statistics | Total: {total} | Pos: {pos/total:.1%} | Neg: {neg/total:.1%} | Zero: {zero/total:.1%}")
                logger.success(f"Flat sample arrays written ({total} points, float32, {len(counts)} objects).")
            shutil.rmtree(parts_dir, ignore_errors=True)

        np.save(intermediate_folder / "idx_str2int_dict.npy", idx_str2int_dict)
        np.save(intermediate_folder / "idx_int2str_dict.npy", idx_int2str_dict)
        np.save(intermediate_folder / "idx_guid2int_dict.npy", idx_guid2int_dict)
        np.save(intermediate_folder / "idx_class_name2idx_dict.npy", class_name_to_idx)
        np.save(intermediate_folder / "idx_class_idx2name_dict.npy", class_idx_to_name)

        # Precise embedding manifest: obj_idx (== embedding row) -> exact source.
        # Used by the write-back step to place each latent into the ONE IFC/GUID it
        # came from, instead of matching every GUID against every IFC file.
        manifest = {
            str(obj_idx): {
                "guid": rec["guid"],
                "source_ifc": rec.get("source_ifc"),
                "ifc_class": rec["ifc_class"],
            }
            for obj_idx, rec in meta_dict.items()
        }
        manifest_path = intermediate_folder / "samples_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.success(f"Sample embedding manifest written to: {manifest_path.name} ({len(manifest)} objects)")

    # ===== Rust sampler backend (sdf_sampler/ crate) =====

    _RUST_CRATE = Path(__file__).resolve().parent.parent / "sdf_sampler"

    @classmethod
    def _rust_sampler_binary(cls) -> Optional[Path]:
        """
        Returns the sdf_sampler release binary, (re)building it with cargo when the
        source is newer. Returns None when unavailable (no crate, no cargo toolchain,
        or build failure) -- callers then fall back to the Python sampler.
        """
        crate = cls._RUST_CRATE
        src = crate / "src" / "main.rs"
        manifest = crate / "Cargo.toml"
        binary = crate / "target" / "release" / "sdf_sampler"
        if not (src.exists() and manifest.exists()):
            return None
        if binary.exists():
            bin_mtime = binary.stat().st_mtime
            if bin_mtime >= src.stat().st_mtime and bin_mtime >= manifest.stat().st_mtime:
                return binary
        if shutil.which("cargo") is None:
            logger.warning("sdf_sampler crate found but cargo is not installed; using Python sampler.")
            return None
        logger.info("Building Rust sampler: cargo build --release ...")
        try:
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=crate, check=True, capture_output=True, text=True, timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err = getattr(e, "stderr", "") or ""
            logger.warning(f"Rust sampler build failed; using Python sampler.\n{err[-2000:]}")
            return None
        return binary if binary.exists() else None

    def _run_rust_sampler(self, jobs, out_dir: Path) -> Optional[dict]:
        """
        Samples `jobs` = [(obj_idx, obj_path_str), ...] with the Rust sampler.
        Writes samples_points/sdf/obj_idx.npy + samples_stats.json into out_dir and
        returns the parsed stats dict, or None if the Rust backend is unavailable
        or the run failed (caller falls back to Python).
        """
        binary = self._rust_sampler_binary()
        if binary is None:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        scale = float(self.config.extraction.scale)
        radii = [float(level.radius) * scale for level in self.config.sampling.levels]
        nums = [int(level.count) for level in self.config.sampling.levels]
        t_half = float(getattr(self.config.sampling, 'min_wall_thickness', 0.1)) * scale / 2.0
        seed = int(getattr(
            getattr(getattr(self.config, "training", None), "reproducibility", None),
            "seed", 42))

        jobs_path = out_dir / "rust_jobs.tsv"
        with open(jobs_path, "w", encoding="utf-8") as f:
            for obj_idx, path in jobs:
                f.write(f"{int(obj_idx)}\t{path}\n")

        cmd = [
            str(binary),
            "--jobs", str(jobs_path),
            "--out", str(out_dir),
            "--seed", str(seed),
            "--threads", str(max(1, os.cpu_count() or 1)),
            "--density", str(int(self.config.sampling.density_per_m2)),
            "--min-per-surface", str(int(self.config.sampling.min_per_surface)),
            "--t-half", repr(t_half),
            "--volume-box", "2.4",
            "--radii", ",".join(repr(r) for r in radii),
            "--nums", ",".join(str(n) for n in nums),
        ]
        logger.info(f"Running Rust sampler on {len(jobs)} objects...")
        try:
            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    logger.info(line)
            ret = proc.wait()
        except OSError as e:
            logger.warning(f"Could not run Rust sampler ({e}); using Python sampler.")
            return None
        finally:
            jobs_path.unlink(missing_ok=True)
        if ret != 0:
            logger.warning(f"Rust sampler exited with code {ret}; using Python sampler.")
            return None
        try:
            with open(out_dir / "samples_stats.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Rust sampler stats unreadable ({e}); using Python sampler.")
            return None

    def sample_single_ifc_subset(
        self, work_folder: Path, source_ifc: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str], int]:
        """
        Re-samples ONLY the objects belonging to `source_ifc` from the OBJs exported
        during preprocessing, using the CURRENT config.sampling settings. Leaves
        samples_dict.npy untouched. Used by the single-IFC hash test (menu 2.3) so
        sampling-density experiments don't require re-running the full pipeline.

        Object indices match the global pipeline enumeration (position in the
        name-sorted OBJ list), so latents/reconstructions stay comparable.

        Returns: (all_indices, all_points, all_sdfs, int2str_map_subset, num_instances_total)
        """
        intermediate_folder = Path(work_folder) / "intermediate"
        objects_dir = intermediate_folder / "objects"
        index_map_path = intermediate_folder / "obj_index_to_ifc.json"
        fix_hint = ("run preprocessing (menu 1.1/1.2) first, or copy intermediate/objects/ + "
                    "obj_index_to_ifc.json from a previous run with identical extraction settings")
        if not objects_dir.is_dir() or not any(objects_dir.glob("*.obj")):
            raise FileNotFoundError(f"No exported OBJs in {objects_dir} -- {fix_hint}")
        if not index_map_path.exists():
            raise FileNotFoundError(f"OBJ index -> IFC map missing: {index_map_path} -- {fix_hint}")
        with open(index_map_path, "r", encoding="utf-8") as f:
            index2ifc = json.load(f)

        sorted_obj_files = sorted(objects_dir.glob("*.obj"), key=lambda p: p.name)

        scale = self.config.extraction.scale
        radii = [level.radius * scale for level in self.config.sampling.levels]
        nums = [level.count for level in self.config.sampling.levels]
        min_per_face = int(self.config.sampling.min_per_surface)
        density_per_m2 = int(self.config.sampling.density_per_m2)
        min_thickness = float(getattr(self.config.sampling, 'min_wall_thickness', 0.1))
        logger.info(
            f"Sampling subset of '{source_ifc}' with density_per_m2={density_per_m2}, "
            f"min_per_surface={min_per_face}"
        )

        int2str_subset: Dict[int, str] = {}
        indices_list, points_list, sdfs_list = [], [], []
        subset = [
            (obj_idx, obj_path)
            for obj_idx, obj_path in enumerate(sorted_obj_files)
            if len(obj_path.stem.split("_")) >= 4
            and index2ifc.get(str(obj_path.stem.split("_")[1])) == source_ifc
        ]
        if not subset:
            avail = sorted(set(index2ifc.values()))
            raise ValueError(f"No OBJs from '{source_ifc}'. Available IFCs: {avail}")

        backend = str(getattr(self.config.sampling, "backend", "rust")).lower()
        if backend == "rust":
            tmp_out = intermediate_folder / "hash_test_rust_out"
            stats = self._run_rust_sampler([(i, str(p)) for i, p in subset], tmp_out)
            if stats is not None:
                all_indices = np.load(tmp_out / "samples_obj_idx.npy")
                all_points = np.load(tmp_out / "samples_points.npy")
                all_sdfs = np.load(tmp_out / "samples_sdf.npy")
                sampled_ids = {int(k) for k, _ in stats["counts"]}
                int2str_subset = {i: p.stem for i, p in subset if i in sampled_ids}
                shutil.rmtree(tmp_out, ignore_errors=True)
                logger.success(
                    f"Sampled {len(int2str_subset)} objects of '{source_ifc}': "
                    f"{all_points.shape[0]} points. (Rust backend)"
                )
                return all_indices, all_points, all_sdfs, int2str_subset, len(sorted_obj_files)
            logger.warning("Falling back to the Python sampler.")

        for obj_idx, obj_path in tqdm(subset, desc=f"Sampling {source_ifc}"):
            try:
                mesh_original = trimesh.load(obj_path, force="mesh", process=False)
                mesh_original = self._clean_mesh(mesh_original)
                if mesh_original is None:
                    continue
                if not mesh_original.is_watertight:
                    logger.warning(f"Repairing non-watertight mesh: {obj_path}")
                    repair.fill_holes(mesh_original)
                    repair.fix_normals(mesh_original)
                assert mesh_original.is_watertight, f"CRITICAL: Mesh {obj_path} is NOT watertight."

                obj_total_area = self._compute_triangle_areas(
                    mesh_original.vertices, mesh_original.faces).sum()
                p_total, sdf, _ = self._sample_points_and_compute_sdf(
                    mesh=mesh_original,
                    total_area=obj_total_area,
                    scale=scale,
                    radii=radii,
                    nums=nums,
                    min_thickness=min_thickness,
                    density_per_m2=density_per_m2,
                    min_per_surface=min_per_face
                )
                if p_total.size == 0:
                    continue
                indices_list.append(np.full(sdf.shape[0], obj_idx, dtype=np.int32))
                points_list.append(p_total.astype(np.float32))
                sdfs_list.append(sdf.astype(np.float32))
                int2str_subset[obj_idx] = obj_path.stem
            except Exception as e:
                logger.error(f"Failed processing {obj_path}: {e}")
                continue

        if not points_list:
            raise ValueError(f"Sampling produced no points for '{source_ifc}'.")

        all_indices = np.concatenate(indices_list, axis=0)
        all_points = np.concatenate(points_list, axis=0)
        all_sdfs = np.concatenate(sdfs_list, axis=0)
        logger.success(
            f"Sampled {len(int2str_subset)} objects of '{source_ifc}': {all_points.shape[0]} points."
        )
        return all_indices, all_points, all_sdfs, int2str_subset, len(sorted_obj_files)

    def _clean_mesh(self, mesh: trimesh.Trimesh, min_face_area: float = 1e-8, min_total_area: float = 1e-6) -> Optional[trimesh.Trimesh]:
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return None
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        repair.fill_holes(mesh)
        repair.fix_normals(mesh)
        if hasattr(mesh, "area_faces"):
            areas = mesh.area_faces
            keep = areas > min_face_area
            if keep.sum() == 0:
                return None
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()
        if mesh.area < min_total_area:
            return None
        return mesh

    def _get_sharp_edge_points(self, mesh: trimesh.Trimesh, num_points: int, angle_threshold: float = np.pi/4) -> np.ndarray:
        try:
            angles = mesh.face_adjacency_angles
            edges_v_indices = mesh.face_adjacency_edges
        except Exception as e:
            logger.warning(f"Could not compute face adjacency: {e}")
            return np.empty((0, 3))
            
        sharp_mask = angles > angle_threshold
        if not np.any(sharp_mask):
            return np.empty((0, 3))
            
        sharp_v_indices = edges_v_indices[sharp_mask]
        v0 = mesh.vertices[sharp_v_indices[:, 0]]
        v1 = mesh.vertices[sharp_v_indices[:, 1]]
        
        edge_lengths = np.linalg.norm(v1 - v0, axis=1)
        total_len = edge_lengths.sum()
        if total_len < 1e-9:
            return np.empty((0, 3))
        
        edge_probs = edge_lengths / total_len
        chosen_indices = np.random.choice(len(v0), size=num_points, p=edge_probs)
        t = np.random.rand(num_points, 1)
        points = v0[chosen_indices] * (1 - t) + v1[chosen_indices] * t
        return points

    def _combine_sample_latent(self, samples: np.ndarray, latent_class: np.ndarray, is_edge: np.ndarray) -> np.ndarray:
        latent_class_full = np.tile(latent_class, (samples.shape[0], 1))
        is_edge_col = is_edge[:, None]
        return np.hstack((latent_class_full, samples, is_edge_col))

    def _compute_triangle_areas(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        return 0.5 * np.linalg.norm(cross_prod, axis=1)

    def _jitter_points_nearby(self, points: np.ndarray, scale: float, radii: list, nums: list, min_thickness: float) -> np.ndarray:
        N = len(points)
        if N == 0:
            return points

        all_offsets = []
        t_half = (min_thickness * scale) / 2.0
        
        for radius, n in zip(radii, nums):
            if n <= 0 or radius <= 0:
                continue

            offset_dirs = np.random.normal(size=(N * n, 3))
            offset_dirs /= (np.linalg.norm(offset_dirs, axis=1, keepdims=True) + 1e-12)

            raw_mags = np.random.uniform(0, radius, size=(N * n, 1))
            # The thickness clamp only applies to sub-thickness levels (keeps fine
            # jitter from crossing a thin wall). Coarser levels ARE the mid/far-field
            # supervision; clamping them collapsed every level into the same ~5mm
            # shell and left the reconstruction volume unsupervised (ghost blobs).
            if radius <= t_half:
                offset_mags = np.clip(raw_mags, 0, t_half * 0.95)
            else:
                offset_mags = raw_mags
            
            base_points = np.repeat(points, n, axis=0)
            new_points = base_points + offset_dirs * offset_mags
            all_offsets.append(new_points)

        if all_offsets:
            return np.vstack(all_offsets + [points])
        return points

    def _sample_points_and_compute_sdf(
        self,
        mesh: trimesh.Trimesh,
        total_area: float,
        scale: float,
        radii: list,
        nums: list,
        min_thickness: float,
        density_per_m2: float,
        min_per_surface: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        
        if faces is None or len(faces) == 0:
            return np.empty((0, 3)), np.empty((0,)), np.empty((0,))

        n_target = int(float(total_area) * density_per_m2)
        n_target = max(n_target, min_per_surface * len(faces))

        num_seeds = max(1, int(n_target * 0.8))
        fid, bc = pcu.sample_mesh_random(verts, faces, num_seeds)
        p_seeds = pcu.interpolate_barycentric_coords(faces, fid, bc, verts)
        
        p_jittered = self._jitter_points_nearby(p_seeds, scale, radii, nums, min_thickness)

        n_vol = int(p_jittered.shape[0] * 0.2)
        p_volume = np.random.uniform(-2.4, 2.4, (n_vol, 3))

        p_total = np.vstack([p_jittered, p_volume])
        is_edge = np.zeros(p_total.shape[0], dtype=np.float32)
        
        sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)
        return p_total, sdf, is_edge

    # ===== Private APIs: Stage 4 (Graph Export) =====

    def _stage_export_graphs(self, work_folder: Path) -> None:
        """
        Extracts spatial topology graphs from IFC files (reads from embedded_ifcs if present,
        otherwise falls back to expanded_ifcs) and saves them to output/.
        """
        logger.info("Exporting spatial topology graphs to JSON...")
        
        # Check for embedded_ifcs first, fallback to expanded_ifcs
        embedded_folder = work_folder / "intermediate" / "embedded_ifcs"
        expanded_folder = work_folder / "intermediate" / "expanded_ifcs"
        
        if embedded_folder.exists() and any(embedded_folder.glob("*.ifc")):
            input_folder = embedded_folder
            logger.info(f"Reading IFC files with embedded latents from: {embedded_folder.as_posix()}")
        else:
            input_folder = expanded_folder
            logger.info(f"Reading IFC files from expanded folder: {expanded_folder.as_posix()}")
            
        output_folder = work_folder / "output"
        output_folder.mkdir(parents=True, exist_ok=True)

        ifc_files = list(input_folder.glob("*.ifc"))
        if not ifc_files:
            logger.warning(f"No IFC files found in {input_folder}. Graph export skipped.")
            return
        for ifc_file_path in ifc_files:
            logger.info(f"Processing IFC graph extraction for: {ifc_file_path.name}")
            try:
                model = ifcopenshell.open(str(ifc_file_path))
                nodes_map, links = self._extract_single_ifc_graph(model)
                
                # Serialize and write to JSON file
                output_nodes = list(nodes_map.values())
                
                seen_links = set()
                unique_links = []
                for l in links:
                    key = (l['start'], l['end'], l['type'])
                    if key not in seen_links:
                        seen_links.add(key)
                        l_copy = dict(l)
                        l_copy["source"] = l["start"]
                        l_copy["target"] = l["end"]
                        unique_links.append(l_copy)

                graph_data = {
                    "graphs": [
                        {
                            "nodes": output_nodes,
                            "links": unique_links
                        }
                    ]
                }

                output_json_path = output_folder / f"{ifc_file_path.stem}_graph.json"
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=2)

                logger.success(f"Graph exported successfully to: {output_json_path}")
            except Exception as e:
                logger.error(f"Failed to process IFC file {ifc_file_path.name}: {e}")

    def _extract_single_ifc_graph(self, model) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        spaces = model.by_type("IfcSpace")
        walls = model.by_type("IfcWall")
        columns = model.by_type("IfcColumn")
        slabs = model.by_type("IfcSlab")
        beams = model.by_type("IfcBeam")
        doors = model.by_type("IfcDoor")
        windows = model.by_type("IfcWindow")
        boundaries = model.by_type("IfcRelSpaceBoundary")

        # 1. Outdoors node stable ID matching build_ifc_graph.py
        outdoors_id = 999999

        # 2. Extract geometry and properties for all elements
        nodes_map = {}
        space_data_map = {}

        # Add outdoors node first (flat matching build_ifc_graph.py structure)
        nodes_map[outdoors_id] = {
            "id": outdoors_id,
            "label": "Exterior (室外)",
            "type": "Space",
            "geometry": self._generate_stable_latent(str(outdoors_id)),
            "coordinates": [0.0, 0.0],
            "IsExternal": 2,
            "Name": "Outdoors",
            "ObjectType": "NONE",
            "Tag": "NONE",
            "OverallHeight": 0.0,
            "OverallWidth": 0.0,
            "OperationType": "NONE",
            "PredefinedType": "NONE",
            "Width": 0,
            "Height": 0,
            "pos_index": 0.0,
            "longName": "Exterior Environment (室外大环境)",
            "center": "N/A",
            "vol_bounds": "N/A"
        }

        # Helper for standard properties extraction
        def process_element(elem, class_label):
            guid = elem.GlobalId
            psets = ifcopenshell.util.element.get_psets(elem)
            
            # Check for LatentCode in NeuralSDF property set
            latent_str = psets.get("NeuralSDF", {}).get("LatentCode")
            latent_code = None
            if latent_str:
                try:
                    latent_code = [float(x) for x in latent_str.split(',')]
                except Exception:
                    pass
            
            if not latent_code or len(latent_code) != 128:
                latent_code = self._generate_stable_latent(guid)

            name = getattr(elem, "Name", "NONE") or "NONE"
            # De-identify: Revit instance names end in ":<id>" (plus "_L<len>" on
            # changing-wall variants), giving every element a unique, copy-stable
            # vocab token. Through the Name categorical the model could then solve
            # the window/wall rules by pure lookup instead of geometry (confirmed:
            # 100% holdout on those rules vs chance on the semantic rule). Keep the
            # family/type part only; spaces keep semantic names for the room rules.
            if class_label != "IfcSpace":
                name = re.sub(r':\d+(_L\d+)?$', '', name)
            obj_type = getattr(elem, "ObjectType", "NONE") or "NONE"
            tag = getattr(elem, "Tag", "NONE") or "NONE"
            height = float(getattr(elem, "OverallHeight", 0.0) or 0.0)
            width = float(getattr(elem, "OverallWidth", 0.0) or 0.0)
            op_type = getattr(elem, "OperationType", "NONE") or "NONE"
            pred_type = getattr(elem, "PredefinedType", "NONE") or "NONE"

            is_ext_val = 2
            if class_label == "IfcWall":
                is_ext = psets.get("Pset_WallCommon", {}).get("IsExternal", False)
                is_ext_val = 1 if is_ext else 0

            node_dict = {
                "id": guid,
                "label": class_label,
                "type": obj_type if obj_type != "NONE" else None,
                "geometry": latent_code,
                "coordinates": [0.0, 0.0],
                "IsExternal": is_ext_val,
                "Name": name,
                "ObjectType": obj_type,
                "Tag": tag,
                "OverallHeight": height,
                "OverallWidth": width,
                "OperationType": op_type,
                "PredefinedType": pred_type,
                "Width": int(width) if class_label in ("IfcDoor", "IfcWindow") else 0,
                "Height": int(height) if class_label in ("IfcDoor", "IfcWindow") else 0,
                "pos_index": 0.0,
                "longName": "NONE",
                "center": "NONE",
                "vol_bounds": "NONE"
            }

            nodes_map[guid] = node_dict

        # Process spaces geometrically
        for s in spaces:
            try:
                shape = ifcopenshell.geom.create_shape(settings, s)
                verts = shape.geometry.verts
                pts = []
                for i in range(0, len(verts), 3):
                    pts.append((verts[i], verts[i+1], verts[i+2]))
                    
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                cz = sum(zs) / len(zs)
                
                space_data_map[s.GlobalId] = {
                    'id': s.GlobalId,
                    'name': s.Name or "Unnamed",
                    'points': list(set(pts)),
                    'center': (cx, cy, cz)
                }
                
                process_element(s, "IfcSpace")
            except Exception as e:
                logger.warning(f"Failed to parse space {s.GlobalId} geometry: {e}")
                process_element(s, "IfcSpace")

        # Process other elements
        for w in walls: process_element(w, "IfcWall")
        for c in columns: process_element(c, "IfcColumn")
        for sl in slabs: process_element(sl, "IfcSlab")
        for b in beams: process_element(b, "IfcBeam")
        for d in doors: process_element(d, "IfcDoor")
        for win in windows: process_element(win, "IfcWindow")

        # 3. Build Wall faces for proximity checks
        wall_spaces = defaultdict(set) # Maps wall_guid -> set of space_guids
        wall_is_external = {} # Maps wall_guid -> boolean

        for w in walls:
            guid = w.GlobalId
            psets = ifcopenshell.util.element.get_psets(w)
            is_ext = psets.get("Pset_WallCommon", {}).get("IsExternal", False)
            wall_is_external[guid] = is_ext

            if is_ext:
                wall_spaces[guid].add(outdoors_id)

            try:
                shape = ifcopenshell.geom.create_shape(settings, w)
                verts = shape.geometry.verts
                faces = shape.geometry.faces
                
                triangles = []
                for i in range(0, len(faces), 3):
                    idx1, idx2, idx3 = faces[i], faces[i+1], faces[i+2]
                    p1 = (verts[idx1*3], verts[idx1*3+1], verts[idx1*3+2])
                    p2 = (verts[idx2*3], verts[idx2*3+1], verts[idx2*3+2])
                    p3 = (verts[idx3*3], verts[idx3*3+1], verts[idx3*3+2])
                    
                    ux, uy, uz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
                    vx, vy, vz = p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]
                    nx = uy*vz - uz*vy
                    ny = uz*vx - ux*vz
                    nz = ux*vy - uy*vx
                    
                    length = math.sqrt(nx*nx + ny*ny + nz*nz)
                    if length > 1e-6:
                        nx, ny, nz = nx/length, ny/length, nz/length
                        area = 0.5 * length
                    else:
                        nx, ny, nz, area = 0.0, 0.0, 0.0, 0.0
                        
                    center = (
                        (p1[0] + p2[0] + p3[0]) / 3.0,
                        (p1[1] + p2[1] + p3[1]) / 3.0,
                        (p1[2] + p2[2] + p3[2]) / 3.0
                    )
                    triangles.append({'p1': p1, 'p2': p2, 'p3': p3, 'normal': (nx, ny, nz), 'center': center, 'area': area})
                    
                clusters = []
                for tri in triangles:
                    if tri['area'] < 1e-5:
                        continue
                    n = tri['normal']
                    matched = None
                    for c in clusters:
                        dot = n[0]*c['normal'][0] + n[1]*c['normal'][1] + n[2]*c['normal'][2]
                        if dot > 0.98:
                            matched = c
                            break
                    if matched:
                        matched['triangles'].append(tri)
                        matched['area'] += tri['area']
                    else:
                        clusters.append({'normal': n, 'triangles': [tri], 'area': tri['area']})
                        
                valid_faces = []
                for c in clusters:
                    if c['area'] < 0.05:
                        continue
                    tx_n, ty_n, tz_n = 0.0, 0.0, 0.0
                    tx_c, ty_c, tz_c = 0.0, 0.0, 0.0
                    tw = 0.0
                    for tri in c['triangles']:
                        w_factor = tri['area']
                        tx_n += tri['normal'][0] * w_factor
                        ty_n += tri['normal'][1] * w_factor
                        tz_n += tri['normal'][2] * w_factor
                        tx_c += tri['center'][0] * w_factor
                        ty_c += tri['center'][1] * w_factor
                        tz_c += tri['center'][2] * w_factor
                        tw += w_factor
                        
                    if tw > 0:
                        len_n = math.sqrt(tx_n**2 + ty_n**2 + tz_n**2)
                        norm = (tx_n/len_n, ty_n/len_n, tz_n/len_n) if len_n > 1e-6 else (0.0, 0.0, 0.0)
                        valid_faces.append({
                            'normal': norm,
                            'center': (tx_c/tw, ty_c/tw, tz_c/tw),
                            'area': tw,
                            'triangles': c['triangles']
                        })
                        
                for face in valid_faces:
                    f_norm = face['normal']
                    f_cent = face['center']
                    
                    f_xs = [v for tri in face['triangles'] for v in (tri['p1'][0], tri['p2'][0], tri['p3'][0])]
                    f_ys = [v for tri in face['triangles'] for v in (tri['p1'][1], tri['p2'][1], tri['p3'][1])]
                    f_zs = [v for tri in face['triangles'] for v in (tri['p1'][2], tri['p2'][2], tri['p3'][2])]
                    min_xf, max_xf = min(f_xs), max(f_xs)
                    min_yf, max_yf = min(f_ys), max(f_ys)
                    min_zf, max_zf = min(f_zs), max(f_zs)
                    
                    for s_guid, s_info in space_data_map.items():
                        if 'points' not in s_info:
                            continue
                        min_perp_dist = float('inf')
                        for pt in s_info['points']:
                            px, py, pz = pt
                            if abs(f_norm[0]) > 0.7:
                                in_tangent = (min_yf - 0.1 <= py <= max_yf + 0.1) and (min_zf - 0.1 <= pz <= max_zf + 0.1)
                            elif abs(f_norm[1]) > 0.7:
                                in_tangent = (min_xf - 0.1 <= px <= max_xf + 0.1) and (min_zf - 0.1 <= pz <= max_zf + 0.1)
                            else:
                                in_tangent = (min_xf - 0.1 <= px <= max_xf + 0.1) and (min_yf - 0.1 <= py <= max_yf + 0.1)
                                
                            if in_tangent:
                                vx = px - f_cent[0]
                                vy = py - f_cent[1]
                                vz = pz - f_cent[2]
                                d_perp = abs(vx*f_norm[0] + vy*f_norm[1] + vz*f_norm[2])
                                if d_perp < min_perp_dist:
                                    min_perp_dist = d_perp
                                    
                        if min_perp_dist < 0.05:
                            wall_spaces[guid].add(s_guid)
            except Exception as e:
                logger.warning(f"Planar proximity check failed for wall {guid}: {e}")

        # 4. Doors & Windows relationships
        door_to_wall_map = {}
        window_to_wall_map = {}
        door_connected_spaces = defaultdict(set)
        window_connected_spaces = defaultdict(set)

        # Trace filling relationships in IFC (Element -> Opening -> Wall)
        for fills in model.by_type("IfcRelFillsElement"):
            element = fills.RelatedBuildingElement
            opening = fills.RelatingOpeningElement
            if element and opening:
                for voids in model.by_type("IfcRelVoidsElement"):
                    if voids.RelatedOpeningElement == opening:
                        wall = voids.RelatingBuildingElement
                        if wall and wall.is_a("IfcWall"):
                            if element.is_a("IfcDoor"):
                                door_to_wall_map[element.GlobalId] = wall.GlobalId
                            elif element.is_a("IfcWindow"):
                                window_to_wall_map[element.GlobalId] = wall.GlobalId

        # Trace connectivity via space boundaries
        for b in boundaries:
            element = b.RelatedBuildingElement
            space = b.RelatingSpace
            if element and space and space.is_a("IfcSpace"):
                if element.is_a("IfcDoor"):
                    door_connected_spaces[element.GlobalId].add(space.GlobalId)
                elif element.is_a("IfcWindow"):
                    window_connected_spaces[element.GlobalId].add(space.GlobalId)

        # Fallback proximity checks for doors
        for d in doors:
            guid = d.GlobalId
            d_center, _ = self._get_shape_center_and_bbox(settings, d)
            
            if guid not in door_to_wall_map and d_center:
                min_w_dist = float('inf')
                closest_w_guid = None
                for w in walls:
                    w_cent, _ = self._get_shape_center_and_bbox(settings, w)
                    if w_cent:
                        dist = math.sqrt((d_center[0]-w_cent[0])**2 + (d_center[1]-w_cent[1])**2 + (d_center[2]-w_cent[2])**2)
                        if dist < min_w_dist:
                            min_w_dist = dist
                            closest_w_guid = w.GlobalId
                if min_w_dist < 1.5:
                    door_to_wall_map[guid] = closest_w_guid

            if len(door_connected_spaces[guid]) < 2 and d_center:
                for s_guid, s_info in space_data_map.items():
                    s_cent = s_info['center']
                    s_dist = math.sqrt((d_center[0]-s_cent[0])**2 + (d_center[1]-s_cent[1])**2 + (d_center[2]-s_cent[2])**2)
                    min_v_dist = min(math.sqrt((d_center[0]-p[0])**2 + (d_center[1]-p[1])**2 + (d_center[2]-p[2])**2) for p in s_info['points']) if 'points' in s_info else float('inf')
                    
                    if min_v_dist < 1.0 or s_dist < 3.0:
                        door_connected_spaces[guid].add(s_guid)

                host_wall_guid = door_to_wall_map.get(guid)
                if host_wall_guid and wall_is_external.get(host_wall_guid, False):
                    door_connected_spaces[guid].add(outdoors_id)

        # Fallback proximity checks for windows
        for win in windows:
            guid = win.GlobalId
            win_center, _ = self._get_shape_center_and_bbox(settings, win)
            
            if guid not in window_to_wall_map and win_center:
                min_w_dist = float('inf')
                closest_w_guid = None
                for w in walls:
                    w_cent, _ = self._get_shape_center_and_bbox(settings, w)
                    if w_cent:
                        dist = math.sqrt((win_center[0]-w_cent[0])**2 + (win_center[1]-w_cent[1])**2 + (win_center[2]-w_cent[2])**2)
                        if dist < min_w_dist:
                            min_w_dist = dist
                            closest_w_guid = w.GlobalId
                if min_w_dist < 1.5:
                    window_to_wall_map[guid] = closest_w_guid

            if len(window_connected_spaces[guid]) < 2 and win_center:
                for s_guid, s_info in space_data_map.items():
                    s_cent = s_info['center']
                    s_dist = math.sqrt((win_center[0]-s_cent[0])**2 + (win_center[1]-s_cent[1])**2 + (win_center[2]-s_cent[2])**2)
                    min_v_dist = min(math.sqrt((win_center[0]-p[0])**2 + (win_center[1]-p[1])**2 + (win_center[2]-p[2])**2) for p in s_info['points']) if 'points' in s_info else float('inf')
                    
                    if min_v_dist < 1.0 or s_dist < 3.0:
                        window_connected_spaces[guid].add(s_guid)

            host_wall_guid = window_to_wall_map.get(guid)
            if host_wall_guid and wall_is_external.get(host_wall_guid, False):
                window_connected_spaces[guid].add(outdoors_id)

        # Calculate position indices and absolute coordinates for all elements
        wall_by_guid = {w.GlobalId: w for w in walls}
        
        # 1. Update coordinates of spaces (using their centers)
        for s_guid, s_info in space_data_map.items():
            if s_guid in nodes_map:
                center = s_info.get('center')
                if center:
                    nodes_map[s_guid]['coordinates'] = [float(center[0]), float(center[1])]
        
        # 2. Update coordinates of walls (using their bottom centerline centers)
        for w in walls:
            guid = w.GlobalId
            if guid in nodes_map:
                try:
                    settings_local = ifcopenshell.geom.settings()
                    settings_local.set(settings_local.USE_WORLD_COORDS, False)
                    wall_shape = ifcopenshell.geom.create_shape(settings_local, w)
                    if wall_shape:
                        wall_verts_local = np.array(wall_shape.geometry.verts).reshape(-1, 3)
                        if len(wall_verts_local) > 0:
                            min_x, max_x = np.min(wall_verts_local[:, 0]), np.max(wall_verts_local[:, 0])
                            z_bottom = np.min(wall_verts_local[:, 2])
                            T_wall = np.asarray(wall_shape.transformation.matrix, dtype=float).reshape(4, 4).T
                            P_local_center = np.array([(min_x + max_x) / 2.0, 0.0, z_bottom, 1.0])
                            P_world_center = T_wall @ P_local_center
                            nodes_map[guid]['coordinates'] = [float(P_world_center[0]), float(P_world_center[1])]
                            continue
                except Exception:
                    pass
                
                # Fallback to general center
                center, _ = self._get_shape_center_and_bbox(settings, w)
                if center:
                    nodes_map[guid]['coordinates'] = [float(center[0]), float(center[1])]

        # 3. Update coordinates of other elements (columns, slabs, beams, etc.)
        other_elements = columns + slabs + beams
        for elem in other_elements:
            guid = elem.GlobalId
            if guid in nodes_map:
                center, _ = self._get_shape_center_and_bbox(settings, elem)
                if center:
                    nodes_map[guid]['coordinates'] = [float(center[0]), float(center[1])]

        # 4. Update coordinates and pos_index of doors and windows.
        # Plan centroid (mean of wall centerline midpoints) anchors the axis-sign
        # canonicalization in _calculate_door_window_properties.
        _wall_coords = [nodes_map[w.GlobalId]["coordinates"] for w in walls
                        if w.GlobalId in nodes_map and nodes_map[w.GlobalId].get("coordinates") not in (None, [0.0, 0.0])]
        plan_centroid = (
            [float(np.mean([c[0] for c in _wall_coords])), float(np.mean([c[1] for c in _wall_coords]))]
            if _wall_coords else None
        )

        for d in doors:
            guid = d.GlobalId
            host_wall_guid = door_to_wall_map.get(guid)
            if host_wall_guid and host_wall_guid in wall_by_guid:
                host_wall = wall_by_guid[host_wall_guid]
                pos_index, coords, wall_center, wall_axis, wall_s = self._calculate_door_window_properties(settings, host_wall, d, plan_centroid)
                if guid in nodes_map:
                    nodes_map[guid]["pos_index"] = pos_index
                    nodes_map[guid]["coordinates"] = coords
                    nodes_map[guid]["host_wall_center"] = wall_center
                    nodes_map[guid]["host_wall_axis"] = wall_axis
                    nodes_map[guid]["wall_s"] = wall_s
            else:
                center, _ = self._get_shape_center_and_bbox(settings, d)
                if center and guid in nodes_map:
                    nodes_map[guid]["coordinates"] = [float(center[0]), float(center[1])]

        for win in windows:
            guid = win.GlobalId
            host_wall_guid = window_to_wall_map.get(guid)
            if host_wall_guid and host_wall_guid in wall_by_guid:
                host_wall = wall_by_guid[host_wall_guid]
                pos_index, coords, wall_center, wall_axis, wall_s = self._calculate_door_window_properties(settings, host_wall, win, plan_centroid)
                if guid in nodes_map:
                    nodes_map[guid]["pos_index"] = pos_index
                    nodes_map[guid]["coordinates"] = coords
                    nodes_map[guid]["host_wall_center"] = wall_center
                    nodes_map[guid]["host_wall_axis"] = wall_axis
                    nodes_map[guid]["wall_s"] = wall_s
            else:
                center, _ = self._get_shape_center_and_bbox(settings, win)
                if center and guid in nodes_map:
                    nodes_map[guid]["coordinates"] = [float(center[0]), float(center[1])]

        # 5. Build links
        links = []
        
        for w_guid, space_guids in wall_spaces.items():
            space_list = list(space_guids)
            if len(space_list) >= 2:
                for s_guid in space_list:
                    links.append({
                        'start': w_guid,
                        'end': s_guid,
                        'type': 'SEPARATES'
                    })
            elif len(space_list) == 1:
                s_guid = space_list[0]
                links.append({
                    'start': w_guid,
                    'end': s_guid,
                    'type': 'INTERNAL_TO'
                })

        for d_guid, space_guids in door_connected_spaces.items():
            for s_guid in space_guids:
                links.append({
                    'start': d_guid,
                    'end': s_guid,
                    'type': 'CONNECTS'
                })

        for win_guid, space_guids in window_connected_spaces.items():
            for s_guid in space_guids:
                links.append({
                    'start': win_guid,
                    'end': s_guid,
                    'type': 'CONNECTS'
                })

        for d_guid, w_guid in door_to_wall_map.items():
            links.append({
                'start': w_guid,
                'end': d_guid,
                'type': 'HAS_DOOR'
            })

        for win_guid, w_guid in window_to_wall_map.items():
            links.append({
                'start': w_guid,
                'end': win_guid,
                'type': 'HAS_WINDOW'
            })

        return nodes_map, links

    @classmethod
    def generate_wall_variants_ifc(cls, work_folder: Path) -> None:
        """
        Step 1: Generate changing wall variants (lengths 100 to 3000 in steps of 100)
        and merge them into a single IFC file `wall_variants.ifc` under input/input_ifcs/.
        """
        work_folder = Path(work_folder)
        input_ifcs_dir = work_folder / "input" / "input_ifcs"
        base_ifc_path = input_ifcs_dir / "Data_base_1.ifc"
        if not base_ifc_path.exists():
            base_ifc_path = Path("data/input/input_ifcs/Data_base_1.ifc")
        if not base_ifc_path.exists():
            raise FileNotFoundError(f"Data_base_1.ifc not found at {base_ifc_path}")
            
        logger.info(f"Generating wall variants IFC from: {base_ifc_path}")
        model = ifcopenshell.open(str(base_ifc_path))
        
        # Target elements GUIDs
        LEFT_WALL_GUID = "02414yPEv25fF9z0uqxlni"
        RIGHT_WALL_GUID = "3U2rKl8q5DVfhs17a5SE44"
        
        left_wall = model.by_guid(LEFT_WALL_GUID)
        right_wall = model.by_guid(RIGHT_WALL_GUID)
        
        if not left_wall or not right_wall:
            raise ValueError("Target walls not found in base IFC file.")
            
        # Explicitly tag the original walls and doors with their original GUIDs so they are preserved
        left_wall.Tag = LEFT_WALL_GUID
        right_wall.Tag = RIGHT_WALL_GUID
        
        # Also tag doors
        LEFT_DOOR_GUID = "0XVTLR_9H3Ox9MAKGyWioT"
        RIGHT_DOOR_GUID = "0i$VJnJOTDM92a$t1VH2bb"
        left_door = model.by_guid(LEFT_DOOR_GUID)
        right_door = model.by_guid(RIGHT_DOOR_GUID)
        if left_door:
            left_door.Tag = LEFT_DOOR_GUID
        if right_door:
            right_door.Tag = RIGHT_DOOR_GUID

        # Tag the six far-end windows with their original GUIDs so the combinatorial
        # graph step can identify them across augmented copies (GUIDs are regenerated
        # per copy; Tags survive). Order = left to right (see instructions.md).
        for win_guid in (
            "2eYTHUu89788SZxgL5ozO4", "2eYTHUu89788SZxgL5ozLa", "2eYTHUu89788SZxgL5ozP6",
            "2eYTHUu89788SZxgL5ozUb", "2eYTHUu89788SZxgL5ozQu", "2eYTHUu89788SZxgL5ozSJ",
        ):
            win = model.by_guid(win_guid)
            if win:
                win.Tag = win_guid
            else:
                logger.warning(f"Far-end window {win_guid} not found in base IFC.")
            
        # Duplication helper
        def duplicate_wall(wall, new_length):
            axis_rep = next(r for r in wall.Representation.Representations if r.RepresentationIdentifier == 'Axis')
            body_rep = next(r for r in wall.Representation.Representations if r.RepresentationIdentifier == 'Body')
            
            # Axis representation
            orig_axis_curve = axis_rep.Items[0]
            orig_axis_pts = orig_axis_curve.Points
            axis_coords = [list(c) for c in orig_axis_pts.CoordList]
            axis_coords[1][0] = float(new_length)
            new_axis_pts = model.create_entity(
                'IfcCartesianPointList3D',
                CoordList=axis_coords
            )
            new_axis_curve = model.create_entity(
                'IfcIndexedPolyCurve',
                Points=new_axis_pts,
                Segments=orig_axis_curve.Segments,
                SelfIntersect=orig_axis_curve.SelfIntersect
            )
            new_axis_rep = model.create_entity(
                'IfcShapeRepresentation',
                ContextOfItems=axis_rep.ContextOfItems,
                RepresentationIdentifier='Axis',
                RepresentationType=axis_rep.RepresentationType,
                Items=[new_axis_curve]
            )
            
            # Body representation
            orig_extruded = body_rep.Items[0]
            orig_swept_area = orig_extruded.SweptArea
            orig_outer_curve = orig_swept_area.OuterCurve
            orig_profile_pts = orig_outer_curve.Points
            profile_coords = [list(c) for c in orig_profile_pts.CoordList]
            profile_coords[0][0] = float(new_length)
            profile_coords[1][0] = float(new_length)
            profile_coords[4][0] = float(new_length)
            
            new_profile_pts = model.create_entity(
                'IfcCartesianPointList2D',
                CoordList=profile_coords
            )
            new_outer_curve = model.create_entity(
                'IfcIndexedPolyCurve',
                Points=new_profile_pts,
                Segments=orig_outer_curve.Segments,
                SelfIntersect=orig_outer_curve.SelfIntersect
            )
            new_swept_area = model.create_entity(
                'IfcArbitraryClosedProfileDef',
                ProfileType=orig_swept_area.ProfileType,
                ProfileName=orig_swept_area.ProfileName,
                OuterCurve=new_outer_curve
            )
            new_extruded = model.create_entity(
                'IfcExtrudedAreaSolid',
                SweptArea=new_swept_area,
                Position=orig_extruded.Position,
                ExtrudedDirection=orig_extruded.ExtrudedDirection,
                Depth=orig_extruded.Depth
            )
            new_body_rep = model.create_entity(
                'IfcShapeRepresentation',
                ContextOfItems=body_rep.ContextOfItems,
                RepresentationIdentifier='Body',
                RepresentationType=body_rep.RepresentationType,
                Items=[new_extruded]
            )
            
            new_shape = model.create_entity(
                'IfcProductDefinitionShape',
                Representations=[new_axis_rep, new_body_rep]
            )
            
            new_guid = ifcopenshell.guid.new()
            new_wall = model.create_entity(
                'IfcWall',
                GlobalId=new_guid,
                OwnerHistory=wall.OwnerHistory,
                Name=f"{wall.Name}_L{new_length}",
                Description=wall.Description,
                ObjectType=wall.ObjectType,
                ObjectPlacement=wall.ObjectPlacement,
                Representation=new_shape,
                Tag=f"{wall.GlobalId}_L{new_length}",
                PredefinedType=wall.PredefinedType
            )
            
            containment = next((rel for rel in model.by_type('IfcRelContainedInSpatialStructure') if wall in rel.RelatedElements), None)
            if containment:
                containment.RelatedElements = list(containment.RelatedElements) + [new_wall]
                
            return new_wall

        # Generate wall variants for Left and Right walls.
        # Step/max come from the run config (extraction.wall_length_step/_max) so the
        # variant IFC and the combinatorial graph loop stay consistent by construction.
        step, max_len = cls._wall_length_params_from_workfolder(work_folder)
        lengths = range(step, max_len + 1, step)
        logger.info(f"Wall variant lengths: {step}..{max_len} step {step} ({len(list(lengths))} per side)")
        for length in range(step, max_len + 1, step):
            duplicate_wall(left_wall, length)
            duplicate_wall(right_wall, length)
            
        output_path = input_ifcs_dir / "wall_variants.ifc"
        model.write(str(output_path))
        logger.success(f"Wall variants merged successfully into: {output_path}")
        
        # Clean up input_ifcs so that only wall_variants.ifc remains for standard preprocessing!
        # We rename other IFCs to .bak
        for item in input_ifcs_dir.iterdir():
            if item.is_file() and item.suffix == ".ifc" and item.name != "wall_variants.ifc":
                bak_path = item.with_suffix(".ifc.bak")
                item.rename(bak_path)
                logger.info(f"Renamed {item.name} to {bak_path.name} to exclude from preprocessing.")

    @staticmethod
    def _wall_length_params_from_workfolder(work_folder: Path) -> Tuple[int, int]:
        """Reads extraction.wall_length_step/_max from the run's config.yaml (defaults 100/3000)."""
        step, max_len = 100, 3000
        cfg_path = Path(work_folder) / "input" / "input_config" / "config.yaml"
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            ext = data.get("extraction", {}) or {}
            step = int(ext.get("wall_length_step", step))
            max_len = int(ext.get("wall_length_max", max_len))
        except Exception:
            logger.warning(f"Could not read wall length params from {cfg_path}; using {step}/{max_len}.")
        return step, max_len

    @classmethod
    def generate_combinatorial_graphs(cls, config_path: Path, work_folder: Path) -> None:
        processor = cls.config_load(config_path)
        try:
            processor._run_combinatorial_graphs(work_folder)
        finally:
            del processor
            logger.info("DataPreprocessor instance has been released.")

    def _run_combinatorial_graphs(self, work_folder: Path) -> None:
        work_folder = Path(work_folder)
        logger.info(f"=== Starting Combinatorial Graph Generation in: {work_folder} ===")
        
        # 1. Scan intermediate/embedded_ifcs/ for IFC files
        embedded_folder = work_folder / "intermediate" / "embedded_ifcs"
        if not embedded_folder.exists() or not any(embedded_folder.glob("*.ifc")):
            logger.error(f"No embedded IFC files found in: {embedded_folder}")
            return
            
        output_folder = work_folder / "output"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Held-out copies: the LAST `holdout_copies` augmentation copies (rigid transforms of
        # the whole plan) are written to output/holdout/ and excluded from the 60/20/20 split,
        # forming an unseen-transform test set. Training scans output/graph_room*.json (non-recursive)
        # so the subfolder is automatically excluded.
        holdout_folder = output_folder / "holdout"
        holdout_folder.mkdir(parents=True, exist_ok=True)

        # Clear old graph files to avoid pollution
        for p in output_folder.glob("graph_room*.json"):
            p.unlink()
        for p in holdout_folder.glob("graph_room*.json"):
            p.unlink()

        ifc_files = sorted(list(embedded_folder.glob("*.ifc")))
        n_holdout = int(getattr(self.config.extraction, "holdout_copies", 0) or 0)
        n_holdout = max(0, min(n_holdout, len(ifc_files) - 1))  # keep at least one copy for training
        holdout_aug_ids = set(range(len(ifc_files) - n_holdout + 1, len(ifc_files) + 1)) if n_holdout else set()
        logger.info(
            f"Generating combinatorial graphs from {len(ifc_files)} rotated/translated IFC files... "
            f"(holding out {n_holdout} copy/copies as unseen test: aug ids {sorted(holdout_aug_ids) or 'none'})"
        )
        
        # Room states config mapping Name to new Name
        room_states_config = {
            1: {
                "100": "Toilet",
                "101": "Store",
                "102": "Bedroom",
                "103": "LivingRoom",
                "104": "Store",
                "105": "Store",
            },
            2: {
                "100": "Store",
                "101": "Toilet",
                "102": "Bedroom",
                "103": "LivingRoom",
                "104": "Store",
                "105": "Store",
            },
            3: {
                "100": "Store",
                "101": "Store",
                "102": "Bedroom",
                "103": "LivingRoom",
                "104": "Toilet",
                "105": "Store",
            },
            4: {
                "100": "Store",
                "101": "Store",
                "102": "Bedroom",
                "103": "LivingRoom",
                "104": "Store",
                "105": "Toilet",
            }
        }
        
        # Target elements GUIDs/tags
        LEFT_WALL_GUID = "02414yPEv25fF9z0uqxlni"
        RIGHT_WALL_GUID = "3U2rKl8q5DVfhs17a5SE44"
        LEFT_DOOR_GUID = "0XVTLR_9H3Ox9MAKGyWioT"
        RIGHT_DOOR_GUID = "0i$VJnJOTDM92a$t1VH2bb"
        
        # Six far-end windows, LEFT TO RIGHT (see instructions.md). Matched by Tag:
        # new variant IFCs carry the GUID as Tag; older exports carry the Revit numeric tag.
        WINDOW_TAGS = [
            ("2eYTHUu89788SZxgL5ozO4", "1667718"),
            ("2eYTHUu89788SZxgL5ozLa", "1667558"),
            ("2eYTHUu89788SZxgL5ozP6", "1667780"),
            ("2eYTHUu89788SZxgL5ozUb", "1667879"),
            ("2eYTHUu89788SZxgL5ozQu", "1667642"),
            ("2eYTHUu89788SZxgL5ozSJ", "1667985"),
        ]
        SYNTH_DOOR_ID = "TARGET_DOOR_SYNTH"
        WIN_TYPE1, WIN_TYPE2 = "WIN_TYPE1", "WIN_TYPE2"
        NUM_POSITIONS = 6

        def determine_door_position(room_state: int, has_left: bool, len_left: int, has_right: bool, len_right: int) -> Optional[int]:
            """
            Wall/room rules -> door position index 1 (leftmost) or 6 (rightmost).
            Returns None when NEITHER changing wall is EFFECTIVE (= present AND
            longer than 1 m): those combos are governed by the six-window rule
            instead (caller enumerates the 6 window states).

            2026-07-11 priority bump: previously only the single both-ABSENT combo
            fell to the window rule, so 6-slot supervision was ~5% of the dataset
            (middle slots ~3%) and the Graphormer plateaued at the predict-the-mean
            loss. Sub-1m walls cannot decide the door anyway (they used to fall back
            to room semantics), so all absent/sub-1m combos now use the window rule,
            lifting the 6-slot share to ~48% of emitted graphs at step=300.
            """
            left_eff = has_left and len_left > 1000
            right_eff = has_right and len_right > 1000

            if not left_eff and not right_eff:
                return None  # six-window rule takes over (absent or sub-1m on both sides)

            # Exactly one effective wall -> door on that side.
            if left_eff and not right_eff:
                return 1
            if right_eff and not left_eff:
                return NUM_POSITIONS

            # Both walls effective -> toilet side decides (room semantics).
            if room_state in (1, 2):
                return 1 if room_state == 1 else NUM_POSITIONS
            return 1 if room_state == 3 else NUM_POSITIONS
                        
        _step = int(getattr(self.config.extraction, "wall_length_step", 100) or 100)
        _maxl = int(getattr(self.config.extraction, "wall_length_max", 3000) or 3000)
        wall_lengths = list(range(_step, _maxl + 1, _step))
        wall_options = [None] + wall_lengths
        logger.info(f"Wall length options per side: {len(wall_options)} (step {_step}, max {_maxl}, +Absent)")
        
        total_graphs = 0
        holdout_graphs = 0

        for aug_idx, ifc_path in enumerate(ifc_files, start=1):
            is_holdout = aug_idx in holdout_aug_ids
            dest_folder = holdout_folder if is_holdout else output_folder
            logger.info(
                f"Extracting base graph template from: {ifc_path.name}"
                f"{' [HELD-OUT unseen test]' if is_holdout else ''}"
            )
            try:
                nodes_map, links = self._extract_single_ifc_graph(ifcopenshell.open(str(ifc_path)))
            except Exception as e:
                logger.error(f"Failed to extract graph from {ifc_path.name}: {e}")
                continue

            # ---- Synthesis context for this copy -------------------------------------
            # The two original target doors serve only as GEOMETRY TEMPLATES (host wall
            # anchor/axis + the two extreme along-wall positions). They are always
            # REMOVED from the emitted graphs and replaced by ONE synthetic empty door
            # whose position is the rule-determined prediction target. This avoids the
            # old leakage where the kept door's own space-links / Name identified its side.
            left_tpl = right_tpl = None
            left_key = right_key = None
            win_keys: List[Optional[str]] = [None] * len(WINDOW_TAGS)
            for k, n in nodes_map.items():
                tag = n.get("Tag") or ""
                if tag == LEFT_DOOR_GUID:
                    left_tpl, left_key = n, k
                elif tag == RIGHT_DOOR_GUID:
                    right_tpl, right_key = n, k
                else:
                    for wi, (wguid, wnum) in enumerate(WINDOW_TAGS):
                        if tag == wguid or tag == wnum:
                            win_keys[wi] = k
            if left_tpl is None or right_tpl is None:
                logger.error(f"Target door templates not found in {ifc_path.name}; skipping copy.")
                continue
            windows_ok = all(win_keys)
            if not windows_ok:
                logger.warning(
                    f"{ifc_path.name}: only {sum(1 for k in win_keys if k)}/6 far-end windows "
                    f"identified; no-effective-wall graphs fall back to room-semantic rule."
                )

            wc = list(left_tpl.get("host_wall_center") or left_tpl.get("coordinates") or [0.0, 0.0])
            axis = list(left_tpl.get("host_wall_axis") or [1.0, 0.0])
            s_left = float(left_tpl.get("wall_s") or 0.0)
            s_right = float(right_tpl.get("wall_s") or 0.0)
            host_wall_key = next(
                (l.get("start") for l in links if l.get("type") == "HAS_DOOR" and l.get("end") == left_key),
                None,
            )
            # Spaces adjacent to the host wall: position-INDEPENDENT link targets for the
            # synthetic door (using the true per-position spaces would leak the answer).
            host_wall_space_keys = [
                (l.get("end") if l.get("start") == host_wall_key else l.get("start"))
                for l in links
                if l.get("type") in ("SEPARATES", "INTERNAL_TO")
                and host_wall_key in (l.get("start"), l.get("end"))
            ]

            # Six evenly spaced positions between the two original door positions.
            pi_l = float(left_tpl.get("pos_index") or 0.0)
            pi_r = float(right_tpl.get("pos_index") or 1.0)
            slots = []
            for k in range(NUM_POSITIONS):
                t = k / (NUM_POSITIONS - 1)
                s_val = s_left + t * (s_right - s_left)
                slots.append({
                    "s": s_val,
                    "coords": [wc[0] + s_val * axis[0], wc[1] + s_val * axis[1]],
                    "pos_index": pi_l + t * (pi_r - pi_l),
                })
            # Align slot order with the WINDOW left->right direction so that slot 1 is on
            # window 1's side in every rotated/translated copy (rotation-covariant).
            # NOTE: the spatial direction of increasing slot index is sign(s_right-s_left)*axis
            # (the s sequence may be decreasing), NOT the raw axis.
            if windows_ok:
                w1c = nodes_map[win_keys[0]].get("coordinates") or [0.0, 0.0]
                w6c = nodes_map[win_keys[-1]].get("coordinates") or [1.0, 0.0]
                u = [w6c[0] - w1c[0], w6c[1] - w1c[1]]
                ds = s_right - s_left
                slot_dir = [ds * axis[0], ds * axis[1]]
                if (slot_dir[0] * u[0] + slot_dir[1] * u[1]) < 0:
                    slots.reverse()

            def emit_variant(room_state: int, len_left, len_right, w_state: int, position: int) -> None:
                """Writes one graph JSON: wall selection + window types + synthetic door at `position` (1..6)."""
                nonlocal total_graphs, holdout_graphs
                has_left = (len_left is not None)
                has_right = (len_right is not None)
                space_config = room_states_config[room_state]

                deleted = {left_key, right_key}
                for node_guid, node in nodes_map.items():
                    tag = node.get("Tag") or ""
                    if tag.startswith(LEFT_WALL_GUID):
                        if not has_left or tag != f"{LEFT_WALL_GUID}_L{len_left}":
                            deleted.add(node_guid)
                    elif tag.startswith(RIGHT_WALL_GUID):
                        if not has_right or tag != f"{RIGHT_WALL_GUID}_L{len_right}":
                            deleted.add(node_guid)

                new_nodes = []
                for node_guid, node_orig in nodes_map.items():
                    if node_guid in deleted:
                        continue
                    node = dict(node_orig)
                    lab = node.get("label")
                    name = node.get("Name")
                    if lab == "IfcSpace" and name in space_config:
                        node["Name"] = space_config[name]
                    # Window typing: exactly one far-end window is TYPE2, the rest TYPE1.
                    if windows_ok and node_guid in win_keys:
                        wi = win_keys.index(node_guid)
                        wtype = WIN_TYPE2 if (wi == w_state - 1) else WIN_TYPE1
                        node["ObjectType"] = wtype
                        node["type"] = wtype
                    new_nodes.append(node)

                # Synthetic empty target door at the rule-determined slot.
                slot = slots[position - 1]
                door_node = dict(left_tpl)
                door_node.update({
                    "id": SYNTH_DOOR_ID,
                    "Tag": SYNTH_DOOR_ID,
                    "Name": "TargetDoorPlaceholder",   # constant: the template door names differ L/R and would leak
                    "geometry": [0.0] * 128,
                    "coordinates": [float(slot["coords"][0]), float(slot["coords"][1])],
                    "pos_index": float(slot["pos_index"]),
                    "host_wall_center": [float(wc[0]), float(wc[1])],
                    "host_wall_axis": [float(axis[0]), float(axis[1])],
                    "wall_s": float(slot["s"]),
                })
                new_nodes.append(door_node)

                new_links = []
                for link in links:
                    start_id = link.get("start")
                    end_id = link.get("end")
                    if start_id in deleted or end_id in deleted:
                        continue
                    new_link = dict(link)
                    new_link["source"] = new_link["start"]
                    new_link["target"] = new_link["end"]
                    new_links.append(new_link)
                # Position-independent connectivity for the synthetic door.
                if host_wall_key and host_wall_key not in deleted:
                    new_links.append({"start": host_wall_key, "end": SYNTH_DOOR_ID, "type": "HAS_DOOR",
                                      "source": host_wall_key, "target": SYNTH_DOOR_ID})
                for sk in host_wall_space_keys:
                    if sk and sk not in deleted:
                        new_links.append({"start": SYNTH_DOOR_ID, "end": sk, "type": "CONNECTS",
                                          "source": SYNTH_DOOR_ID, "target": sk})

                l_str = f"L{len_left}" if has_left else "LAbsent"
                r_str = f"R{len_right}" if has_right else "RAbsent"
                filename = f"graph_room{room_state}_aug{aug_idx}_{l_str}_{r_str}_W{w_state}_graph.json"
                with open(dest_folder / filename, "w", encoding="utf-8") as out_f:
                    json.dump({"graphs": [{"nodes": new_nodes, "links": new_links}]}, out_f,
                              ensure_ascii=False, indent=2)
                if is_holdout:
                    holdout_graphs += 1
                else:
                    total_graphs += 1

            # ---- Combinatorial loop for this copy ------------------------------------
            for room_state in (1, 2, 3, 4):
                for len_left in wall_options:
                    has_left = (len_left is not None)
                    for len_right in wall_options:
                        has_right = (len_right is not None)
                        position = determine_door_position(
                            room_state, has_left, len_left or 0, has_right, len_right or 0
                        )
                        if position is None and windows_ok:
                            # Six-window rule (no effective wall on either side): enumerate
                            # all 6 window states; the door sits at the COMPLEMENTARY
                            # position of the TYPE2 window (w -> 7-w).
                            for w_state in range(1, NUM_POSITIONS + 1):
                                emit_variant(room_state, len_left, len_right,
                                             w_state, NUM_POSITIONS + 1 - w_state)
                            continue
                        if position is None:
                            # Windows unavailable: fall back to the legacy room-semantic rule.
                            if room_state in (1, 2):
                                position = 1 if room_state == 1 else NUM_POSITIONS
                            else:
                                position = 1 if room_state == 3 else NUM_POSITIONS
                        # Wall-rule case: window types are irrelevant to the answer, but they
                        # must still VARY (deterministically) so the model cannot form a
                        # spurious "type2 position <-> door position" shortcut outside the
                        # both-absent case.
                        w_state = 1 + zlib.crc32(
                            f"{aug_idx}|{room_state}|{len_left}|{len_right}".encode()
                        ) % NUM_POSITIONS
                        emit_variant(room_state, len_left, len_right, w_state, position)

        logger.success(
            f"Generated {total_graphs} combinatorial graph files under output/"
            f" and {holdout_graphs} held-out graph files under output/holdout/."
        )

    def _generate_stable_latent(self, guid: str) -> List[float]:
        """Generates a stable 128-dimensional latent vector seeded by element GUID."""
        import hashlib
        md5_hash = hashlib.md5(str(guid).encode('utf-8')).hexdigest()
        seed = int(md5_hash[:8], 16) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        return rng.random(128, dtype=np.float32).tolist()

    def _get_shape_center_and_bbox(self, settings, element) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float, float, float, float]]]:
        """Calculates geometry center and bbox for an element."""
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
            verts = shape.geometry.verts
            xs = [verts[i] for i in range(0, len(verts), 3)]
            ys = [verts[i+1] for i in range(0, len(verts), 3)]
            zs = [verts[i+2] for i in range(0, len(verts), 3)]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            cz = sum(zs) / len(zs)
            return (cx, cy, cz), (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
        except Exception:
            return None, None

    # ===== Private APIs: Directory Management Utility =====

    def _clear_folder(self, folder_path: Path) -> None:
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Path does not exist: {folder_path}")
            return
        if not path.is_dir():
            logger.warning(f"Path is not a directory: {folder_path}")
            return
        for item in path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink() 
                elif item.is_dir():
                    shutil.rmtree(item) 
            except Exception as e:
                logger.error(f"Failed to delete {item}: {e}")
        logger.success(f"Folder cleared: {folder_path}")

    def _copy_to_folder(self, source_folder_path: Path, target_folder_path: Path) -> None:
        source_path = Path(source_folder_path)
        target_path = Path(target_folder_path)
        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_folder_path}")
            return
        if not source_path.is_dir():
            logger.warning(f"Source path is not a directory: {source_folder_path}")
            return
        target_path.mkdir(parents=True, exist_ok=True)
        for item in source_path.iterdir():
            dest = target_path / item.name
            try:
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to copy {item} to {dest}: {e}")

    def _clear_previous_sample_data(self, target_folder: Path) -> None:
        folder_path = Path(target_folder)
        file_list = [
            "samples_dict.npy",
            "samples_points.npy",
            "samples_sdf.npy",
            "samples_obj_idx.npy",
            "idx_str2int_dict.npy",
            "idx_int2str_dict.npy",
            "idx_guid2int_dict.npy",
            "idx_class_name2idx_dict.npy",
            "idx_class_idx2name_dict.npy"
        ]
        if all(not (folder_path / f).exists() for f in file_list):
            logger.info("No previous files exist, skipping...")
            return
        else:
            for file_name in file_list:
                file_path = folder_path / file_name
                file_path.unlink(missing_ok=True)
            logger.success(f"Previous sampling data in {target_folder} has been deleted.")

    def _calculate_door_window_properties(self, settings, host_wall, element, plan_centroid=None) -> Tuple[float, List[float], List[float], List[float], float]:
        """
        Calculates properties of a door/window relative to its host wall.

        Returns a 5-tuple:
            pos_index          : relative center position along wall length (0.0 to 1.0)
            coordinates        : absolute 2D [X, Y] of the element center on the wall's bottom centerline
            host_wall_center   : absolute 2D [X, Y] of the WALL's centerline midpoint (public anchor)
            host_wall_axis     : 2D unit vector = world direction of the wall's intrinsic local +x
                                 (a door-independent, rotation-covariant orientation; used to give the
                                 signed along-wall scalar `s` a stable sign so left/right never flips)
            s                  : signed along-wall offset in world meters such that
                                 coordinates == host_wall_center + s * host_wall_axis

        This is the anchor/axis/offset decomposition for the Plan-B wall-relative
        coordinate head: door_world = wall_center + s * axis.
        """
        _FALLBACK = (0.0, [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], 0.0)
        try:
            # Create local settings object to get geometry in local system
            settings_local = ifcopenshell.geom.settings()
            settings_local.set(settings_local.USE_WORLD_COORDS, False)

            # Create world settings object to get geometry in world coordinates
            settings_world = ifcopenshell.geom.settings()
            settings_world.set(settings_world.USE_WORLD_COORDS, True)

            # Get wall shape in local coordinates
            wall_shape = ifcopenshell.geom.create_shape(settings_local, host_wall)
            if not wall_shape or not hasattr(wall_shape, 'geometry') or not hasattr(wall_shape, 'transformation'):
                return _FALLBACK

            wall_verts_local = np.array(wall_shape.geometry.verts).reshape(-1, 3)
            if len(wall_verts_local) == 0:
                return _FALLBACK

            min_x, max_x = np.min(wall_verts_local[:, 0]), np.max(wall_verts_local[:, 0])
            z_bottom = np.min(wall_verts_local[:, 2])
            wall_len = max_x - min_x
            if wall_len < 1e-6:
                wall_len = 1e-6

            T_wall = np.asarray(wall_shape.transformation.matrix, dtype=float).reshape(4, 4).T

            # Get door/window shape in world coordinates
            elem_shape = ifcopenshell.geom.create_shape(settings_world, element)
            if not elem_shape or not hasattr(elem_shape, 'geometry'):
                return _FALLBACK

            elem_verts_world = np.array(elem_shape.geometry.verts).reshape(-1, 3)
            if len(elem_verts_world) == 0:
                return _FALLBACK

            # Transform door/window vertices into wall local coordinates
            elem_verts_hom = np.c_[elem_verts_world, np.ones(len(elem_verts_world))]
            try:
                T_wall_inv = np.linalg.inv(T_wall)
            except np.linalg.LinAlgError:
                T_wall_inv = np.linalg.pinv(T_wall)

            elem_verts_local = (T_wall_inv @ elem_verts_hom.T).T[:, :3]

            elem_min_x, elem_max_x = np.min(elem_verts_local[:, 0]), np.max(elem_verts_local[:, 0])
            elem_center_x = (elem_min_x + elem_max_x) / 2.0

            rel_center = (elem_center_x - min_x) / wall_len
            rel_center = float(np.clip(rel_center, 0.0, 1.0))

            # Project onto the wall's bottom centerline
            P_local_proj = np.array([elem_center_x, 0.0, z_bottom, 1.0])
            P_world_proj = T_wall @ P_local_proj
            coords = [float(P_world_proj[0]), float(P_world_proj[1])]

            # ----- Plan-B anchor / axis / signed offset -----
            # Wall centerline midpoint (public anchor, independent of the door).
            center_x_local = (min_x + max_x) / 2.0
            P_center_world = T_wall @ np.array([center_x_local, 0.0, z_bottom, 1.0])
            wall_center = [float(P_center_world[0]), float(P_center_world[1])]

            # World direction of the wall's intrinsic local +x (length) axis.
            dir_world = T_wall @ np.array([1.0, 0.0, 0.0, 0.0])
            axis_xy = dir_world[:2]
            n = float(np.linalg.norm(axis_xy))
            axis_unit = (axis_xy / n) if n > 1e-9 else np.array([1.0, 0.0])

            # Canonicalize the axis SIGN so it no longer depends on the wall's arbitrary
            # modeling direction (a flipped local +x would silently flip the sign of `s`,
            # which downstream IS the left/right label). Rule: orient the axis so the plan
            # centroid lies to its LEFT (cross(axis, centroid - wall_center).z > 0). The
            # rule is rotation-covariant (centroid and axis rotate together with the copy)
            # and purely geometric, so re-modeled/re-exported walls get the same sign.
            if plan_centroid is not None:
                _wc_tmp = T_wall @ np.array([(min_x + max_x) / 2.0, 0.0, z_bottom, 1.0])
                v = np.array([plan_centroid[0] - _wc_tmp[0], plan_centroid[1] - _wc_tmp[1]])
                cross_z = axis_unit[0] * v[1] - axis_unit[1] * v[0]
                if abs(cross_z) > 1e-9 and cross_z < 0:
                    axis_unit = -axis_unit

            # Signed along-wall offset so that coords == wall_center + s * axis_unit.
            s = float(np.dot(np.array(coords) - np.array(wall_center), axis_unit))

            return rel_center, coords, wall_center, [float(axis_unit[0]), float(axis_unit[1])], s

        except Exception as e:
            logger.warning(f"Failed to calculate door/window properties for {element.GlobalId} on wall {host_wall.GlobalId}: {e}")
            return _FALLBACK

