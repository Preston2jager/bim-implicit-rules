//! SDF sample generator for the SDF_3 preprocessing pipeline (stage 3).
//!
//! Mirrors DataPreprocessor._sample_points_and_compute_sdf / _jitter_points_nearby:
//!   - area-weighted surface seed sampling (0.8 * n_target seeds)
//!   - multi-level Gaussian-direction jitter; the thickness clamp applies ONLY to
//!     levels with radius <= t_half (coarse levels are the mid/far-field supervision)
//!   - 20% uniform volume samples in [-volume_box, volume_box]^3
//!   - signed distance via BVH closest-point + angle-weighted pseudo-normals
//!     (Baerentzen & Aanaes), after enforcing consistent outward winding
//!
//! Std-only (no crates.io), multithreaded over objects. Writes the flat arrays the
//! trainers consume directly:
//!   <out>/samples_points.npy  float32 (N,3)
//!   <out>/samples_sdf.npy     float32 (N,)
//!   <out>/samples_obj_idx.npy int32   (N,)
//!   <out>/samples_stats.json  counts / sign stats / bbox
//!
//! Usage (all args required except --threads):
//!   sdf_sampler --jobs jobs.tsv --out DIR --seed 42 --density 10000 \
//!       --min-per-surface 1000 --t-half 0.005 --radii 0.001,0.005,... \
//!       --nums 12,10,8,4,4 --volume-box 2.4 [--threads 16]
//! jobs.tsv lines: "<obj_idx>\t<path to .obj>"

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

type V3 = [f64; 3];

#[inline]
fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
#[inline]
fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
#[inline]
fn scale(a: V3, s: f64) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
#[inline]
fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
fn norm(a: V3) -> f64 { dot(a, a).sqrt() }
#[inline]
fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n > 1e-30 { scale(a, 1.0 / n) } else { [0.0, 0.0, 0.0] }
}

// ============================= RNG (SplitMix64) =============================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    #[inline]
    fn f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0) }
    /// Standard normal (Box-Muller, one value per call for simplicity).
    #[inline]
    fn normal(&mut self) -> f64 {
        let u1 = (1.0 - self.f64()).max(1e-300); // avoid ln(0)
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ============================= OBJ loading =============================

struct Mesh {
    v: Vec<V3>,
    f: Vec<[u32; 3]>,
}

fn parse_obj(path: &Path) -> std::io::Result<Mesh> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(1 << 16, file);
    let mut v: Vec<V3> = Vec::new();
    let mut f: Vec<[u32; 3]> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let bytes = line.as_bytes();
        if bytes.len() < 2 {
            continue;
        }
        if bytes[0] == b'v' && bytes[1] == b' ' {
            let mut it = line[2..].split_ascii_whitespace();
            let x: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let y: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let z: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            if x.is_finite() && y.is_finite() && z.is_finite() {
                v.push([x, y, z]);
            } else {
                v.push([0.0, 0.0, 0.0]);
            }
        } else if bytes[0] == b'f' && bytes[1] == b' ' {
            // "f a b c ..." possibly with /vt/vn suffixes; fan-triangulate polygons.
            let idx: Vec<u32> = line[2..]
                .split_ascii_whitespace()
                .filter_map(|tok| tok.split('/').next())
                .filter_map(|s| s.parse::<i64>().ok())
                .map(|i| if i > 0 { (i - 1) as u32 } else { (v.len() as i64 + i) as u32 })
                .collect();
            for k in 2..idx.len() {
                f.push([idx[0], idx[k - 1], idx[k]]);
            }
        }
    }
    Ok(Mesh { v, f })
}

fn tri_area(v0: V3, v1: V3, v2: V3) -> f64 {
    norm(cross(sub(v1, v0), sub(v2, v0))) * 0.5
}

/// Mirror of _clean_mesh: drop degenerate faces (area <= 1e-8); reject meshes with
/// total area < 1e-6. Returns per-face areas alongside the cleaned mesh.
fn clean_mesh(mut mesh: Mesh) -> Option<(Mesh, Vec<f64>, f64)> {
    if mesh.v.is_empty() || mesh.f.is_empty() {
        return None;
    }
    let nv = mesh.v.len() as u32;
    mesh.f.retain(|t| t[0] < nv && t[1] < nv && t[2] < nv);
    let mut areas: Vec<f64> = Vec::with_capacity(mesh.f.len());
    let mut kept: Vec<[u32; 3]> = Vec::with_capacity(mesh.f.len());
    let mut total = 0.0;
    for t in &mesh.f {
        let a = tri_area(mesh.v[t[0] as usize], mesh.v[t[1] as usize], mesh.v[t[2] as usize]);
        if a > 1e-8 {
            kept.push(*t);
            areas.push(a);
            total += a;
        }
    }
    if kept.is_empty() || total < 1e-6 {
        return None;
    }
    mesh.f = kept;
    Some((mesh, areas, total))
}

/// Enforce consistent winding by BFS over shared edges, then flip everything if the
/// signed volume says the surface points inward. Pseudo-normal signs need this.
fn orient_mesh(mesh: &mut Mesh) {
    let nf = mesh.f.len();
    // edge (min,max) -> up to 2 (face, reversed?) entries
    let mut edge_map: HashMap<(u32, u32), Vec<(u32, bool)>> = HashMap::with_capacity(nf * 3);
    for (fi, t) in mesh.f.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            let key = (a.min(b), a.max(b));
            edge_map.entry(key).or_default().push((fi as u32, a > b));
        }
    }
    let mut flipped = vec![false; nf];
    let mut visited = vec![false; nf];
    let mut stack: Vec<u32> = Vec::new();
    for start in 0..nf {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        stack.push(start as u32);
        while let Some(fi) = stack.pop() {
            let t = mesh.f[fi as usize];
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let key = (a.min(b), a.max(b));
                let this_rev = (a > b) != flipped[fi as usize];
                if let Some(list) = edge_map.get(&key) {
                    for &(fj, rev_j) in list {
                        if fj == fi || visited[fj as usize] {
                            continue;
                        }
                        // Consistent winding = the two faces traverse the shared
                        // edge in opposite directions.
                        let neighbor_rev = rev_j != flipped[fj as usize];
                        if neighbor_rev == this_rev {
                            flipped[fj as usize] = !flipped[fj as usize];
                        }
                        visited[fj as usize] = true;
                        stack.push(fj);
                    }
                }
            }
        }
    }
    for (fi, t) in mesh.f.iter_mut().enumerate() {
        if flipped[fi] {
            t.swap(1, 2);
        }
    }
    // Outward check: signed volume of the closed surface should be positive.
    let mut vol6 = 0.0;
    for t in &mesh.f {
        let (a, b, c) = (mesh.v[t[0] as usize], mesh.v[t[1] as usize], mesh.v[t[2] as usize]);
        vol6 += dot(a, cross(b, c));
    }
    if vol6 < 0.0 {
        for t in mesh.f.iter_mut() {
            t.swap(1, 2);
        }
    }
}

// ================== Pseudo-normals (Baerentzen & Aanaes) ==================

struct SignedMesh {
    v: Vec<V3>,
    f: Vec<[u32; 3]>,
    face_n: Vec<V3>,
    vert_n: Vec<V3>,
    edge_n: HashMap<(u32, u32), V3>,
    bvh: Bvh,
}

impl SignedMesh {
    fn build(mesh: Mesh) -> SignedMesh {
        let nf = mesh.f.len();
        let mut face_n: Vec<V3> = Vec::with_capacity(nf);
        let mut vert_n: Vec<V3> = vec![[0.0; 3]; mesh.v.len()];
        let mut edge_n: HashMap<(u32, u32), V3> = HashMap::with_capacity(nf * 3 / 2);
        for t in &mesh.f {
            let (a, b, c) = (mesh.v[t[0] as usize], mesh.v[t[1] as usize], mesh.v[t[2] as usize]);
            let n = normalize(cross(sub(b, a), sub(c, a)));
            face_n.push(n);
            // Angle-weighted vertex pseudo-normals.
            let corners = [(a, b, c, t[0]), (b, c, a, t[1]), (c, a, b, t[2])];
            for (p, q, r, vi) in corners {
                let e1 = normalize(sub(q, p));
                let e2 = normalize(sub(r, p));
                let ang = dot(e1, e2).clamp(-1.0, 1.0).acos();
                vert_n[vi as usize] = add(vert_n[vi as usize], scale(n, ang));
            }
            // Edge pseudo-normals: sum of the (two) adjacent face normals.
            for e in 0..3 {
                let (x, y) = (t[e], t[(e + 1) % 3]);
                let key = (x.min(y), x.max(y));
                let entry = edge_n.entry(key).or_insert([0.0; 3]);
                *entry = add(*entry, n);
            }
        }
        for n in vert_n.iter_mut() {
            *n = normalize(*n);
        }
        for n in edge_n.values_mut() {
            *n = normalize(*n);
        }
        let bvh = Bvh::build(&mesh.v, &mesh.f);
        SignedMesh { v: mesh.v, f: mesh.f, face_n, vert_n, edge_n, bvh }
    }

    /// Signed distance to the mesh at `p`.
    fn signed_distance(&self, p: V3) -> f64 {
        let (dist2, cp, tri) = self.bvh.closest(&self.v, &self.f, p);
        let t = self.f[tri];
        let (a, b, c) = (self.v[t[0] as usize], self.v[t[1] as usize], self.v[t[2] as usize]);
        // Barycentric coordinates of cp to pick the right pseudo-normal.
        let (w0, w1, w2) = barycentric(cp, a, b, c);
        let eps = 1e-6;
        let n = if w0 > eps && w1 > eps && w2 > eps {
            self.face_n[tri]
        } else if w0 <= eps && w1 <= eps {
            self.vert_n[t[2] as usize]
        } else if w0 <= eps && w2 <= eps {
            self.vert_n[t[1] as usize]
        } else if w1 <= eps && w2 <= eps {
            self.vert_n[t[0] as usize]
        } else if w0 <= eps {
            let key = (t[1].min(t[2]), t[1].max(t[2]));
            self.edge_n.get(&key).copied().unwrap_or(self.face_n[tri])
        } else if w1 <= eps {
            let key = (t[0].min(t[2]), t[0].max(t[2]));
            self.edge_n.get(&key).copied().unwrap_or(self.face_n[tri])
        } else {
            let key = (t[0].min(t[1]), t[0].max(t[1]));
            self.edge_n.get(&key).copied().unwrap_or(self.face_n[tri])
        };
        let d = dist2.sqrt();
        if dot(sub(p, cp), n) >= 0.0 { d } else { -d }
    }
}

fn barycentric(p: V3, a: V3, b: V3, c: V3) -> (f64, f64, f64) {
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(p, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 {
        return (1.0, 0.0, 0.0);
    }
    let w1 = (d11 * d20 - d01 * d21) / denom;
    let w2 = (d00 * d21 - d01 * d20) / denom;
    (1.0 - w1 - w2, w1, w2)
}

/// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection).
fn closest_point_triangle(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return add(a, scale(ab, t));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return add(a, scale(ac, t));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), t));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

// ============================= BVH =============================

struct BvhNode {
    bb_min: V3,
    bb_max: V3,
    left: u32,   // internal: left child index; leaf: start into tri_ids
    right: u32,  // internal: right child index; unused for leaves
    count: u32,  // 0 for internal nodes, else leaf triangle count
}

struct Bvh {
    nodes: Vec<BvhNode>,
    tri_ids: Vec<u32>,
}

impl Bvh {
    fn build(v: &[V3], f: &[[u32; 3]]) -> Bvh {
        let nf = f.len();
        let mut centroids: Vec<V3> = Vec::with_capacity(nf);
        let mut boxes: Vec<(V3, V3)> = Vec::with_capacity(nf);
        for t in f {
            let (a, b, c) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
            let mn = [a[0].min(b[0]).min(c[0]), a[1].min(b[1]).min(c[1]), a[2].min(b[2]).min(c[2])];
            let mx = [a[0].max(b[0]).max(c[0]), a[1].max(b[1]).max(c[1]), a[2].max(b[2]).max(c[2])];
            boxes.push((mn, mx));
            centroids.push(scale(add(mn, mx), 0.5));
        }
        let mut tri_ids: Vec<u32> = (0..nf as u32).collect();
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * nf);
        Self::build_node(&mut nodes, &mut tri_ids, &centroids, &boxes, 0, nf);
        Bvh { nodes, tri_ids }
    }

    fn build_node(
        nodes: &mut Vec<BvhNode>,
        tri_ids: &mut [u32],
        centroids: &[V3],
        boxes: &[(V3, V3)],
        start: usize,
        count: usize,
    ) -> u32 {
        let mut bb_min = [f64::INFINITY; 3];
        let mut bb_max = [f64::NEG_INFINITY; 3];
        for &tid in &tri_ids[start..start + count] {
            let (mn, mx) = boxes[tid as usize];
            for d in 0..3 {
                bb_min[d] = bb_min[d].min(mn[d]);
                bb_max[d] = bb_max[d].max(mx[d]);
            }
        }
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode { bb_min, bb_max, left: start as u32, right: 0, count: count as u32 });
        if count <= 8 {
            return node_idx;
        }
        // Median split on the longest axis of the centroid extent.
        let mut ext_min = [f64::INFINITY; 3];
        let mut ext_max = [f64::NEG_INFINITY; 3];
        for &tid in &tri_ids[start..start + count] {
            let cn = centroids[tid as usize];
            for d in 0..3 {
                ext_min[d] = ext_min[d].min(cn[d]);
                ext_max[d] = ext_max[d].max(cn[d]);
            }
        }
        let mut axis = 0;
        let mut best = -1.0;
        for d in 0..3 {
            let e = ext_max[d] - ext_min[d];
            if e > best {
                best = e;
                axis = d;
            }
        }
        if best <= 1e-12 {
            return node_idx; // degenerate spread: keep as (large) leaf
        }
        let mid = count / 2;
        tri_ids[start..start + count].select_nth_unstable_by(mid, |&x, &y| {
            centroids[x as usize][axis]
                .partial_cmp(&centroids[y as usize][axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let left = Self::build_node(nodes, tri_ids, centroids, boxes, start, mid);
        let right = Self::build_node(nodes, tri_ids, centroids, boxes, start + mid, count - mid);
        nodes[node_idx as usize].left = left;
        nodes[node_idx as usize].right = right;
        nodes[node_idx as usize].count = 0;
        node_idx
    }

    #[inline]
    fn aabb_dist2(bb_min: V3, bb_max: V3, p: V3) -> f64 {
        let mut d2 = 0.0;
        for d in 0..3 {
            let v = if p[d] < bb_min[d] {
                bb_min[d] - p[d]
            } else if p[d] > bb_max[d] {
                p[d] - bb_max[d]
            } else {
                0.0
            };
            d2 += v * v;
        }
        d2
    }

    fn closest(&self, v: &[V3], f: &[[u32; 3]], p: V3) -> (f64, V3, usize) {
        let mut best_d2 = f64::INFINITY;
        let mut best_cp = [0.0; 3];
        let mut best_tri = 0usize;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni as usize];
            if Self::aabb_dist2(node.bb_min, node.bb_max, p) >= best_d2 {
                continue;
            }
            if node.count > 0 {
                let start = node.left as usize;
                for &tid in &self.tri_ids[start..start + node.count as usize] {
                    let t = f[tid as usize];
                    let cp = closest_point_triangle(
                        p, v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
                    let d = sub(p, cp);
                    let d2 = dot(d, d);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_cp = cp;
                        best_tri = tid as usize;
                    }
                }
            } else {
                let l = node.left;
                let r = node.right;
                // Visit the nearer child first (better pruning).
                let dl = Self::aabb_dist2(self.nodes[l as usize].bb_min, self.nodes[l as usize].bb_max, p);
                let dr = Self::aabb_dist2(self.nodes[r as usize].bb_min, self.nodes[r as usize].bb_max, p);
                if dl <= dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        (best_d2, best_cp, best_tri)
    }
}

// ============================= Sampling =============================

#[derive(Clone)]
struct Params {
    radii: Vec<f64>, // already scaled (config radius * extraction scale)
    nums: Vec<usize>,
    density_per_m2: f64,
    min_per_surface: usize,
    t_half: f64, // min_wall_thickness * scale / 2
    volume_box: f64,
    seed: u64,
}

struct ObjResult {
    obj_idx: u32,
    n: u64,
    pos: u64,
    neg: u64,
    bb_min: V3,
    bb_max: V3,
}

fn sample_object(
    sm: &SignedMesh,
    areas: &[f64],
    total_area: f64,
    params: &Params,
    rng: &mut Rng,
) -> (Vec<f32>, Vec<f32>) {
    // Cumulative area table for area-weighted face picking.
    let mut cum: Vec<f64> = Vec::with_capacity(areas.len());
    let mut acc = 0.0;
    for &a in areas {
        acc += a;
        cum.push(acc);
    }

    let n_target = ((total_area * params.density_per_m2) as usize)
        .max(params.min_per_surface * sm.f.len());
    let num_seeds = ((n_target as f64) * 0.8).max(1.0) as usize;

    let mut seeds: Vec<V3> = Vec::with_capacity(num_seeds);
    for _ in 0..num_seeds {
        let r = rng.f64() * acc;
        let fi = cum.partition_point(|&c| c < r).min(areas.len() - 1);
        let t = sm.f[fi];
        let (a, b, c) = (sm.v[t[0] as usize], sm.v[t[1] as usize], sm.v[t[2] as usize]);
        // Uniform barycentric (sqrt trick).
        let su = rng.f64().sqrt();
        let bv = rng.f64() * su;
        let au = 1.0 - su;
        let cu = 1.0 - au - bv;
        seeds.push(add(add(scale(a, au), scale(b, bv)), scale(c, cu)));
    }

    let n_jitter: usize = params.nums.iter().sum::<usize>() * num_seeds + num_seeds;
    let n_vol = (n_jitter as f64 * 0.2) as usize;
    let mut pts: Vec<V3> = Vec::with_capacity(n_jitter + n_vol);

    for (&radius, &n) in params.radii.iter().zip(params.nums.iter()) {
        if n == 0 || radius <= 0.0 {
            continue;
        }
        let clamp = radius <= params.t_half;
        for &s in &seeds {
            for _ in 0..n {
                let dir = normalize([rng.normal(), rng.normal(), rng.normal()]);
                let mut mag = rng.f64() * radius;
                if clamp {
                    mag = mag.min(params.t_half * 0.95);
                }
                pts.push(add(s, scale(dir, mag)));
            }
        }
    }
    pts.extend_from_slice(&seeds);
    for _ in 0..n_vol {
        pts.push([
            (rng.f64() * 2.0 - 1.0) * params.volume_box,
            (rng.f64() * 2.0 - 1.0) * params.volume_box,
            (rng.f64() * 2.0 - 1.0) * params.volume_box,
        ]);
    }

    let mut out_pts: Vec<f32> = Vec::with_capacity(pts.len() * 3);
    let mut out_sdf: Vec<f32> = Vec::with_capacity(pts.len());
    for &p in &pts {
        let d = sm.signed_distance(p);
        out_pts.push(p[0] as f32);
        out_pts.push(p[1] as f32);
        out_pts.push(p[2] as f32);
        out_sdf.push(d as f32);
    }
    (out_pts, out_sdf)
}

// ============================= .npy writing =============================

fn write_npy_header<W: Write>(w: &mut W, descr: &str, shape: &str) -> std::io::Result<()> {
    let dict = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", descr, shape);
    let base = 10 + dict.len(); // magic(6)+ver(2)+hlen(2)
    let padded = (base + 63) / 64 * 64;
    let header_len = padded - 10;
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header_len as u16).to_le_bytes())?;
    w.write_all(dict.as_bytes())?;
    for _ in 0..(header_len - dict.len() - 1) {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;
    Ok(())
}

fn f32s_to_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// ============================= main =============================

fn parse_args() -> (PathBuf, PathBuf, Params, usize) {
    let mut jobs_path = None;
    let mut out_dir = None;
    let mut threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut radii: Vec<f64> = vec![];
    let mut nums: Vec<usize> = vec![];
    let mut density = 5000.0;
    let mut min_per_surface = 1000usize;
    let mut t_half = 0.005;
    let mut volume_box = 2.4;
    let mut seed = 42u64;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i + 1 < args.len() {
        let val = &args[i + 1];
        match args[i].as_str() {
            "--jobs" => jobs_path = Some(PathBuf::from(val)),
            "--out" => out_dir = Some(PathBuf::from(val)),
            "--threads" => threads = val.parse().expect("bad --threads"),
            "--radii" => radii = val.split(',').map(|s| s.parse().expect("bad --radii")).collect(),
            "--nums" => nums = val.split(',').map(|s| s.parse().expect("bad --nums")).collect(),
            "--density" => density = val.parse().expect("bad --density"),
            "--min-per-surface" => min_per_surface = val.parse().expect("bad --min-per-surface"),
            "--t-half" => t_half = val.parse().expect("bad --t-half"),
            "--volume-box" => volume_box = val.parse().expect("bad --volume-box"),
            "--seed" => seed = val.parse().expect("bad --seed"),
            other => panic!("unknown argument: {}", other),
        }
        i += 2;
    }
    let jobs_path = jobs_path.expect("--jobs is required");
    let out_dir = out_dir.expect("--out is required");
    assert_eq!(radii.len(), nums.len(), "--radii and --nums length mismatch");
    (
        jobs_path,
        out_dir,
        Params { radii, nums, density_per_m2: density, min_per_surface, t_half, volume_box, seed },
        threads.max(1),
    )
}

fn main() {
    let (jobs_path, out_dir, params, threads) = parse_args();

    let jobs_file = File::open(&jobs_path).expect("cannot open jobs file");
    let mut jobs: Vec<(u32, PathBuf)> = Vec::new();
    for line in BufReader::new(jobs_file).lines() {
        let line = line.expect("jobs read error");
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.splitn(2, '\t');
        let idx: u32 = it.next().unwrap().trim().parse().expect("bad obj_idx in jobs file");
        let path = PathBuf::from(it.next().expect("jobs line missing path").trim());
        jobs.push((idx, path));
    }
    let n_jobs = jobs.len();
    eprintln!("[sdf_sampler] {} objects, {} threads", n_jobs, threads);

    let parts_dir = out_dir.join("rust_parts");
    let _ = fs::remove_dir_all(&parts_dir);
    fs::create_dir_all(&parts_dir).expect("cannot create parts dir");

    let jobs = Arc::new(jobs);
    let next_job = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<ObjResult>>> = Arc::new(Mutex::new(Vec::with_capacity(n_jobs)));

    let mut handles = Vec::new();
    for _ in 0..threads {
        let jobs = Arc::clone(&jobs);
        let next_job = Arc::clone(&next_job);
        let done = Arc::clone(&done);
        let results = Arc::clone(&results);
        let params = params.clone();
        let parts_dir = parts_dir.clone();
        handles.push(thread::spawn(move || loop {
            let j = next_job.fetch_add(1, Ordering::Relaxed);
            if j >= jobs.len() {
                break;
            }
            let (obj_idx, ref path) = jobs[j];
            let mesh = match parse_obj(path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[sdf_sampler] ERROR reading {}: {}", path.display(), e);
                    continue;
                }
            };
            let Some((mut mesh, areas, total_area)) = clean_mesh(mesh) else {
                eprintln!("[sdf_sampler] skipped degenerate mesh: {}", path.display());
                continue;
            };
            orient_mesh(&mut mesh);
            let sm = SignedMesh::build(mesh);
            let mut rng = Rng::new(params.seed.wrapping_add(obj_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03);
            let (pts, sdf) = sample_object(&sm, &areas, total_area, &params, &mut rng);
            if sdf.is_empty() {
                continue;
            }

            let mut bb_min = [f64::INFINITY; 3];
            let mut bb_max = [f64::NEG_INFINITY; 3];
            for ch in pts.chunks_exact(3) {
                for d in 0..3 {
                    bb_min[d] = bb_min[d].min(ch[d] as f64);
                    bb_max[d] = bb_max[d].max(ch[d] as f64);
                }
            }
            let pos = sdf.iter().filter(|&&s| s > 1e-6).count() as u64;
            let neg = sdf.iter().filter(|&&s| s < -1e-6).count() as u64;

            let pts_path = parts_dir.join(format!("{:06}.pts", obj_idx));
            let sdf_path = parts_dir.join(format!("{:06}.sdf", obj_idx));
            fs::write(&pts_path, f32s_to_le_bytes(&pts)).expect("part write failed");
            fs::write(&sdf_path, f32s_to_le_bytes(&sdf)).expect("part write failed");

            results.lock().unwrap().push(ObjResult {
                obj_idx,
                n: sdf.len() as u64,
                pos,
                neg,
                bb_min,
                bb_max,
            });
            let k = done.fetch_add(1, Ordering::Relaxed) + 1;
            if k % 25 == 0 || k == n_jobs {
                eprintln!("[sdf_sampler] progress {}/{}", k, n_jobs);
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }

    let mut results = Arc::try_unwrap(results).ok().unwrap().into_inner().unwrap();
    results.sort_by_key(|r| r.obj_idx);
    let total: u64 = results.iter().map(|r| r.n).sum();
    if total == 0 {
        eprintln!("[sdf_sampler] no points sampled");
        std::process::exit(2);
    }

    // Assemble flat .npy arrays (streamed; one part in memory at a time).
    let mut pts_w = BufWriter::with_capacity(1 << 20, File::create(out_dir.join("samples_points.npy")).unwrap());
    let mut sdf_w = BufWriter::with_capacity(1 << 20, File::create(out_dir.join("samples_sdf.npy")).unwrap());
    let mut idx_w = BufWriter::with_capacity(1 << 20, File::create(out_dir.join("samples_obj_idx.npy")).unwrap());
    write_npy_header(&mut pts_w, "<f4", &format!("({}, 3)", total)).unwrap();
    write_npy_header(&mut sdf_w, "<f4", &format!("({},)", total)).unwrap();
    write_npy_header(&mut idx_w, "<i4", &format!("({},)", total)).unwrap();

    let mut buf: Vec<u8> = Vec::new();
    for r in &results {
        for (suffix, w) in [("pts", &mut pts_w), ("sdf", &mut sdf_w)] {
            let mut fpart = File::open(parts_dir.join(format!("{:06}.{}", r.obj_idx, suffix))).unwrap();
            buf.clear();
            fpart.read_to_end(&mut buf).unwrap();
            w.write_all(&buf).unwrap();
        }
        let idx_bytes = (r.obj_idx as i32).to_le_bytes();
        let mut idx_buf = Vec::with_capacity(r.n as usize * 4);
        for _ in 0..r.n {
            idx_buf.extend_from_slice(&idx_bytes);
        }
        idx_w.write_all(&idx_buf).unwrap();
    }
    pts_w.flush().unwrap();
    sdf_w.flush().unwrap();
    idx_w.flush().unwrap();
    let _ = fs::remove_dir_all(&parts_dir);

    // Global stats (hand-rolled JSON; std-only).
    let pos: u64 = results.iter().map(|r| r.pos).sum();
    let neg: u64 = results.iter().map(|r| r.neg).sum();
    let mut bb_min = [f64::INFINITY; 3];
    let mut bb_max = [f64::NEG_INFINITY; 3];
    for r in &results {
        for d in 0..3 {
            bb_min[d] = bb_min[d].min(r.bb_min[d]);
            bb_max[d] = bb_max[d].max(r.bb_max[d]);
        }
    }
    let counts_json: Vec<String> = results
        .iter()
        .map(|r| format!("[{}, {}]", r.obj_idx, r.n))
        .collect();
    let stats = format!(
        "{{\n  \"total\": {},\n  \"pos\": {},\n  \"neg\": {},\n  \"zero\": {},\n  \"bbox_min\": [{}, {}, {}],\n  \"bbox_max\": [{}, {}, {}],\n  \"objects\": {},\n  \"counts\": [{}]\n}}\n",
        total, pos, neg, total - pos - neg,
        bb_min[0], bb_min[1], bb_min[2], bb_max[0], bb_max[1], bb_max[2],
        results.len(),
        counts_json.join(", ")
    );
    fs::write(out_dir.join("samples_stats.json"), stats).unwrap();
    eprintln!(
        "[sdf_sampler] done: {} points from {} objects (pos {:.1}%, neg {:.1}%)",
        total,
        results.len(),
        pos as f64 / total as f64 * 100.0,
        neg as f64 / total as f64 * 100.0
    );
}
