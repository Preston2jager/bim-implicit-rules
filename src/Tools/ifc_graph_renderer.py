#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IFC Spatial Graph Renderer
==========================
Reads a semantic topology graph JSON (conforming to the graph.json schema)
and renders an interactive, premium HTML visualization dashboard using vis-network and Tailwind CSS.

Usage:
  uv run python src/Tools/ifc_graph_renderer.py -i <graph_json_path> -o <output_html_path>
"""

import os
import sys
import json
import argparse
import webbrowser
from pathlib import Path

# Premium HTML Dashboard Template using vis-network and Tailwind CSS
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IFC Spatial Topology Graph Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #030712;
        }
        .glass-card {
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(51, 65, 85, 0.4);
        }
        #network-container {
            background: radial-gradient(circle, #0f172a 0%, #020617 100%);
            width: 100%;
            height: 100%;
            min-height: 600px;
        }
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }
    </style>
</head>
<body class="min-h-screen text-slate-100 p-4 md:p-6 flex flex-col gap-6">

    <!-- Header -->
    <div class="glass-card p-5 rounded-2xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
            <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 text-[10px] font-semibold rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">Semantic Graph Model</span>
                <span class="text-slate-400 text-xs">IFC Spatial Topology</span>
            </div>
            <h1 class="text-xl md:text-2xl font-bold mt-1.5 bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">BIM Spatial Topology & Connectivity Graph</h1>
            <p class="text-slate-400 text-xs mt-0.5">Model File: <span class="text-indigo-400 font-mono">__FILE_NAME__</span></p>
        </div>
        <div class="text-right text-xs text-slate-400">
            <p>Nodes: <span class="text-indigo-400 font-bold" id="stat-nodes">0</span> | Edges: <span class="text-pink-400 font-bold" id="stat-edges">0</span></p>
            <p class="mt-0.5 text-[10px]">Method: <span class="text-slate-300 font-mono">Data Preprocessor Graph Export</span></p>
        </div>
    </div>

    <!-- Main Workspace -->
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[720px]">
        
        <!-- Interactive Graph Area (Left 3 Columns) -->
        <div class="lg:col-span-3 glass-card rounded-2xl overflow-hidden flex flex-col h-full relative">
            <div class="p-3 border-b border-slate-800 bg-slate-900/30 flex flex-wrap justify-between items-center gap-2 z-10">
                <div class="flex items-center gap-4 text-xs">
                    <span class="font-semibold text-slate-200">Interactive Canvas</span>
                    <div class="flex flex-wrap gap-4 pl-4 border-l border-slate-800 text-[11px] text-slate-400">
                        <div class="flex items-center gap-1.5"><span class="w-3 h-3 bg-purple-500 rounded-full border border-purple-400"></span>Space</div>
                        <div class="flex items-center gap-1.5"><span class="w-3.5 h-2.5 bg-slate-600 rounded border border-slate-400"></span>Wall</div>
                        <div class="flex items-center gap-1.5"><span class="w-2.5 h-3.5 bg-emerald-500 rounded-md border border-emerald-400"></span>Door</div>
                        <div class="flex items-center gap-1.5"><span class="w-2.5 h-3.5 bg-blue-500 rounded-md border border-blue-400"></span>Window</div>
                        <div class="flex items-center gap-1.5"><span class="w-3 h-3 bg-amber-500 rounded border border-amber-400"></span>Others</div>
                    </div>
                </div>
                <div class="flex gap-2">
                    <button onclick="resetView()" class="px-2.5 py-1 text-xs font-semibold rounded bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 border border-indigo-500/20 transition-all">Reset Camera</button>
                    <button onclick="togglePhysics()" id="btn-physics" class="px-2.5 py-1 text-xs font-semibold rounded bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 transition-all">Physics: ON</button>
                </div>
            </div>
            
            <div id="network-container" class="flex-1 w-full" style="min-height: 600px;"></div>
            
            <!-- Edge Type Legend overlay -->
            <div class="absolute bottom-4 left-4 p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-[10px] text-slate-300 space-y-2 z-10">
                <div class="font-bold border-b border-slate-800 pb-1 text-slate-400 uppercase tracking-wider">Edge Relationships</div>
                <div class="flex items-center gap-2"><span class="w-5 h-0.5 bg-pink-500 inline-block"></span><span class="font-semibold text-pink-400">SEPARATES</span> (Wall separates rooms)</div>
                <div class="flex items-center gap-2"><span class="w-5 h-0.5 border-t border-dashed border-orange-400 inline-block"></span><span class="font-semibold text-orange-400">INTERNAL_TO</span> (Wall inside a room)</div>
                <div class="flex items-center gap-2"><span class="w-5 h-0.5 bg-emerald-500 inline-block"></span><span class="font-semibold text-emerald-400">CONNECTS</span> (Door/Window connects room)</div>
                <div class="flex items-center gap-2"><span class="w-5 h-0.5 border-t border-dotted border-slate-400 inline-block"></span><span class="font-semibold text-slate-400">HAS_DOOR / HAS_WINDOW</span> (Wall hosts element)</div>
            </div>
        </div>

        <!-- Sidebar Details & Control Panel (Right 1 Column) -->
        <div class="glass-card rounded-2xl flex flex-col h-full overflow-hidden">
            <!-- Filter Options -->
            <div class="p-4 border-b border-slate-800 bg-slate-900/30 space-y-3">
                <div class="text-xs font-bold uppercase tracking-wider text-slate-400">Filter Elements</div>
                <div class="grid grid-cols-3 gap-2">
                    <label class="flex flex-col items-center p-2 rounded-lg bg-slate-950/40 border border-slate-800 hover:border-slate-700 cursor-pointer">
                        <input type="checkbox" id="chk-space" checked onchange="filterNodes()" class="sr-only peer">
                        <span class="text-[10px] text-slate-400 peer-checked:text-indigo-400 font-semibold">Spaces</span>
                        <span class="w-2 h-2 rounded-full bg-slate-700 peer-checked:bg-indigo-500 mt-1"></span>
                    </label>
                    <label class="flex flex-col items-center p-2 rounded-lg bg-slate-950/40 border border-slate-800 hover:border-slate-700 cursor-pointer">
                        <input type="checkbox" id="chk-wall" checked onchange="filterNodes()" class="sr-only peer">
                        <span class="text-[10px] text-slate-400 peer-checked:text-slate-300 font-semibold">Walls</span>
                        <span class="w-2 h-2 rounded bg-slate-700 peer-checked:bg-slate-400 mt-1"></span>
                    </label>
                    <label class="flex flex-col items-center p-2 rounded-lg bg-slate-950/40 border border-slate-800 hover:border-slate-700 cursor-pointer">
                        <input type="checkbox" id="chk-door" checked onchange="filterNodes()" class="sr-only peer">
                        <span class="text-[10px] text-slate-400 peer-checked:text-emerald-400 font-semibold">Doors</span>
                        <span class="w-2 h-2 rounded bg-slate-700 peer-checked:bg-emerald-500 mt-1"></span>
                    </label>
                    <label class="flex flex-col items-center p-2 rounded-lg bg-slate-950/40 border border-slate-800 hover:border-slate-700 cursor-pointer col-span-2">
                        <input type="checkbox" id="chk-window" checked onchange="filterNodes()" class="sr-only peer">
                        <span class="text-[10px] text-slate-400 peer-checked:text-blue-400 font-semibold">Windows</span>
                        <span class="w-2 h-2 rounded bg-slate-700 peer-checked:bg-blue-500 mt-1"></span>
                    </label>
                    <label class="flex flex-col items-center p-2 rounded-lg bg-slate-950/40 border border-slate-800 hover:border-slate-700 cursor-pointer">
                        <input type="checkbox" id="chk-other" checked onchange="filterNodes()" class="sr-only peer">
                        <span class="text-[10px] text-slate-400 peer-checked:text-amber-400 font-semibold">Others</span>
                        <span class="w-2 h-2 rounded bg-slate-700 peer-checked:bg-amber-500 mt-1"></span>
                    </label>
                </div>
                
                <input type="text" id="search-input" onkeyup="searchNodes()" placeholder="Search elements by ID or name..." class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 outline-none placeholder:text-slate-600 focus:border-indigo-500/50">
            </div>

            <!-- Detail Panel -->
            <div class="flex-1 p-5 overflow-y-auto" id="detail-panel">
                <div class="h-full flex flex-col justify-center items-center text-center text-slate-500 text-xs py-12 space-y-2">
                    <svg class="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"></path></svg>
                    <p>Select any node or edge in the graph canvas to inspect its semantic topology properties.</p>
                </div>
            </div>
            
            <!-- Footer info -->
            <div class="p-3 border-t border-slate-800 bg-slate-950/40 text-[10px] text-slate-500 text-center">
                Interactive Graph Visualizer (vis-network Standalone)
            </div>
        </div>
    </div>

    <!-- Data Injection & Graph Config -->
    <script>
        const graphData = __GRAPH_DATA_JSON__;
        
        let network = null;
        let physicsEnabled = true;
        
        // Initial vis.js datasets
        const rawNodes = new vis.DataSet(graphData.nodes.map(n => {
            let displayName = n.Name || n.type || n.label;
            if (displayName.includes(':')) {
                const parts = displayName.split(':');
                displayName = parts[parts.length - 1]; 
            }
            
            let label = `${n.label}\\n(${displayName})`;
            let color = '#312e81'; 
            let border = '#4f46e5';
            let shape = 'circle';
            let size = 22;
            
            if (n.label === 'IfcWall') {
                color = '#334155';
                border = '#64748b';
                shape = 'box';
                size = 14;
            } else if (n.label === 'IfcDoor') {
                color = '#065f46';
                border = '#10b981';
                shape = 'diamond';
                size = 14;
            } else if (n.label === 'IfcWindow') {
                color = '#1e3a8a';
                border = '#3b82f6';
                shape = 'triangle';
                size = 14;
            } else if (n.label === 'IfcColumn') {
                color = '#78350f';
                border = '#f59e0b';
                shape = 'star';
                size = 14;
            } else if (n.label === 'IfcSlab') {
                color = '#1f2937';
                border = '#4b5563';
                shape = 'ellipse';
                size = 14;
            } else if (n.label === 'IfcBeam') {
                color = '#5b21b6';
                border = '#8b5cf6';
                shape = 'database';
                size = 14;
            }
            
            const properties = {
                'id': n.id,
                'class': n.label,
                'name': n.Name || 'N/A',
                'type': n.type || 'N/A',
                'tag': n.Tag || 'N/A',
                'predefinedType': n.PredefinedType || 'N/A',
                'width': n.OverallWidth || 'N/A',
                'height': n.OverallHeight || 'N/A',
                'operation': n.OperationType || 'N/A'
            };
            
            return {
                id: n.id,
                label: label,
                title: `${n.label}: ${n.Name || n.type || n.id}`,
                color: {
                    background: color,
                    border: border,
                    highlight: { background: border, border: '#ffffff' },
                    hover: { background: border, border: '#ffffff' }
                },
                shape: shape,
                size: size,
                type: n.label,
                properties: properties
            };
        }));
        
        const rawEdges = new vis.DataSet(graphData.links.map((l, idx) => {
            let color = '#ec4899'; // SEPARATES (pink)
            let dashes = false;
            
            if (l.type === 'INTERNAL_TO') {
                color = '#f97316'; // orange dashed
                dashes = [5, 5];
            } else if (l.type === 'CONNECTS') {
                color = '#10b981'; // emerald green
            } else if (l.type === 'HAS_DOOR' || l.type === 'HOSTS') {
                color = '#94a3b8'; // slate grey dotted
                dashes = [2, 6];
            } else if (l.type === 'HAS_WINDOW') {
                color = '#3b82f6'; // blue dotted
                dashes = [2, 4];
            }
            
            const edgeObj = {
                id: idx + 1,
                from: l.start,
                to: l.end,
                label: l.type,
                color: {
                    color: color,
                    highlight: '#ffffff',
                    hover: color
                },
                width: l.type === 'CONNECTS' ? 3 : ((l.type === 'SEPARATES' || l.type === 'HAS_DOOR' || l.type === 'HAS_WINDOW') ? 2 : 1.5),
                dashes: dashes
            };
            
            if (l.type === 'HAS_DOOR' || l.type === 'HAS_WINDOW' || l.type === 'HOSTS') {
                edgeObj.arrows = { to: { enabled: true, scaleFactor: 0.8 } };
            }
            
            return edgeObj;
        }));
        
        // Filter variables
        const viewNodes = new vis.DataView(rawNodes, {
            filter: function (item) {
                const spaceChk = document.getElementById('chk-space').checked;
                const wallChk = document.getElementById('chk-wall').checked;
                const doorChk = document.getElementById('chk-door').checked;
                const windowChk = document.getElementById('chk-window').checked;
                const otherChk = document.getElementById('chk-other').checked;
                
                if (item.type === 'IfcSpace') return spaceChk;
                if (item.type === 'IfcWall') return wallChk;
                if (item.type === 'IfcDoor') return doorChk;
                if (item.type === 'IfcWindow') return windowChk;
                return otherChk; // IfcColumn, IfcSlab, IfcBeam
            }
        });

        // Initialize Network
        function initNetwork() {
            const container = document.getElementById('network-container');
            const data = {
                nodes: viewNodes,
                edges: rawEdges
            };
            
            const options = {
                nodes: {
                    font: { size: 10, face: 'Plus Jakarta Sans', color: '#f8fafc' },
                    borderWidth: 1.5,
                    shadow: { enabled: true, color: 'rgba(0,0,0,0.5)', size: 4, x: 1, y: 1 }
                },
                edges: {
                    font: { size: 8, face: 'Plus Jakarta Sans', align: 'middle', color: '#64748b' },
                    hoverWidth: 1.5,
                    smooth: {
                        type: 'continuous',
                        roundness: 0.2
                    }
                },
                interaction: {
                    hover: true,
                    selectConnectedEdges: true,
                    tooltipDelay: 200
                },
                physics: {
                    barnesHut: {
                        gravitationalConstant: -1800,
                        centralGravity: 0.2,
                        springLength: 120,
                        springConstant: 0.04,
                        damping: 0.09
                    },
                    stabilization: {
                        iterations: 150,
                        fit: true
                    }
                }
            };
            
            network = new vis.Network(container, data, options);
            
            // Set statistics
            document.getElementById('stat-nodes').innerText = rawNodes.length;
            document.getElementById('stat-edges').innerText = rawEdges.length;
            
            // Event Handlers
            network.on("selectNode", function (params) {
                const nodeId = params.nodes[0];
                const node = rawNodes.get(nodeId);
                showNodeDetails(node);
            });
            
            network.on("selectEdge", function (params) {
                if (params.nodes.length === 0) {
                    const edgeId = params.edges[0];
                    const edge = rawEdges.get(edgeId);
                    showEdgeDetails(edge);
                }
            });
            
            network.on("deselectNode", function () {
                clearDetails();
            });
            
            network.on("deselectEdge", function () {
                clearDetails();
            });
        }
        
        // Show details in sidebar
        function showNodeDetails(node) {
            const panel = document.getElementById('detail-panel');
            
            let propsHTML = '';
            for (let [k, v] of Object.entries(node.properties)) {
                propsHTML += `
                    <div class="flex flex-col py-1.5 border-b border-slate-800/50">
                        <span class="text-[9px] font-bold text-slate-500 uppercase tracking-wider">${k}</span>
                        <span class="text-xs text-slate-300 font-mono mt-0.5">${v}</span>
                    </div>
                `;
            }
            
            const connectedEdges = rawEdges.get({
                filter: function (edge) {
                    return edge.from === node.id || edge.to === node.id;
                }
            });
            
            let relationsHTML = '';
            connectedEdges.forEach(e => {
                const otherNodeId = e.from === node.id ? e.to : e.from;
                const otherNode = rawNodes.get(otherNodeId);
                if (!otherNode) return;
                
                let relBadge = '';
                if (e.label === 'CONNECTS') relBadge = 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
                else if (e.label === 'SEPARATES') relBadge = 'bg-pink-500/10 text-pink-400 border border-pink-500/20';
                else if (e.label === 'INTERNAL_TO') relBadge = 'bg-orange-500/10 text-orange-400 border border-orange-500/20';
                else if (e.label === 'HAS_DOOR' || e.label === 'HOSTS') relBadge = 'bg-slate-500/10 text-slate-300 border border-slate-500/20';
                else if (e.label === 'HAS_WINDOW') relBadge = 'bg-blue-500/10 text-blue-400 border border-blue-500/20';
                
                relationsHTML += `
                    <div class="p-2.5 rounded-xl bg-slate-950/40 border border-slate-800/80 hover:border-indigo-500/30 flex items-center justify-between text-[11px] cursor-pointer" onclick="focusOnNode('${otherNode.id}')">
                        <div class="flex flex-col">
                            <span class="font-semibold text-slate-200">${otherNode.type}</span>
                            <span class="text-[9px] text-slate-500">${otherNode.properties.name || 'N/A'}</span>
                        </div>
                        <span class="px-1.5 py-0.5 text-[9px] rounded font-medium ${relBadge}">${e.label}</span>
                    </div>
                `;
            });
            
            let iconColor = 'text-indigo-400';
            if (node.type === 'IfcWall') iconColor = 'text-slate-400';
            if (node.type === 'IfcDoor') iconColor = 'text-emerald-400';
            if (node.type === 'IfcWindow') iconColor = 'text-blue-400';
            
            panel.innerHTML = `
                <div class="space-y-6">
                    <div class="pb-4 border-b border-slate-800">
                        <div class="flex items-center gap-2">
                            <span class="px-2 py-0.5 text-[9px] font-bold uppercase rounded bg-slate-800 ${iconColor}">${node.type}</span>
                            <span class="text-[10px] text-slate-500 font-mono">ID #${node.id.substring(0, 12)}...</span>
                        </div>
                        <h2 class="text-base font-bold text-slate-100 mt-2">${node.properties.name}</h2>
                    </div>
                    
                    <div class="space-y-2">
                        <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400">Attributes</h3>
                        <div class="space-y-1">${propsHTML}</div>
                    </div>
                    
                    <div class="space-y-3">
                        <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400">Topology Connections</h3>
                        <div class="space-y-2 max-h-[200px] overflow-y-auto pr-1">${relationsHTML || '<p class="text-xs italic text-slate-500">No active links</p>'}</div>
                    </div>
                </div>
            `;
        }
        
        function showEdgeDetails(edge) {
            const panel = document.getElementById('detail-panel');
            const fromNode = rawNodes.get(edge.from);
            const toNode = rawNodes.get(edge.to);
            
            panel.innerHTML = `
                <div class="space-y-6">
                    <div class="pb-4 border-b border-slate-800">
                        <div class="flex items-center gap-2">
                            <span class="px-2 py-0.5 text-[9px] font-bold uppercase rounded bg-pink-500/10 text-pink-400 border border-pink-500/20">Relationship</span>
                            <span class="text-[10px] text-slate-500 font-mono">ID ${edge.id}</span>
                        </div>
                        <h2 class="text-base font-bold text-slate-100 mt-2">${edge.label}</h2>
                    </div>
                    
                    <div class="space-y-3 p-3 rounded-xl bg-slate-950/40 border border-slate-800">
                        <div class="flex flex-col">
                            <span class="text-[9px] text-slate-500 uppercase font-semibold">From Source</span>
                            <span class="text-xs font-bold text-indigo-300 hover:underline cursor-pointer" onclick="focusOnNode('${fromNode.id}')">${fromNode.type}</span>
                            <span class="text-[10px] text-slate-400 mt-0.5">${fromNode.properties.name}</span>
                        </div>
                        <div class="border-t border-slate-800 my-1"></div>
                        <div class="flex flex-col">
                            <span class="text-[9px] text-slate-500 uppercase font-semibold">To Target</span>
                            <span class="text-xs font-bold text-indigo-300 hover:underline cursor-pointer" onclick="focusOnNode('${toNode.id}')">${toNode.type}</span>
                            <span class="text-[10px] text-slate-400 mt-0.5">${toNode.properties.name}</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function clearDetails() {
            document.getElementById('detail-panel').innerHTML = `
                <div class="h-full flex flex-col justify-center items-center text-center text-slate-500 text-xs py-12 space-y-2">
                    <svg class="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"></path></svg>
                    <p>Select any node or edge in the graph canvas to inspect its semantic topology properties.</p>
                </div>
            `;
        }
        
        function focusOnNode(nodeId) {
            network.selectNodes([nodeId]);
            const node = rawNodes.get(nodeId);
            showNodeDetails(node);
            network.focus(nodeId, {
                scale: 1.2,
                animation: { duration: 500 }
            });
        }
        
        function resetView() {
            network.fit({ animation: { duration: 500 } });
        }
        
        function togglePhysics() {
            physicsEnabled = !physicsEnabled;
            network.setOptions({ physics: { enabled: physicsEnabled } });
            
            const btn = document.getElementById('btn-physics');
            if (physicsEnabled) {
                btn.innerText = "Physics: ON";
                btn.className = "px-2.5 py-1 text-xs font-semibold rounded bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 transition-all";
            } else {
                btn.innerText = "Physics: OFF";
                btn.className = "px-2.5 py-1 text-xs font-semibold rounded bg-slate-500/10 hover:bg-slate-500/20 text-slate-400 border border-slate-500/20 transition-all";
            }
        }
        
        function filterNodes() {
            viewNodes.refresh();
            resetView();
        }
        
        function searchNodes() {
            const val = document.getElementById('search-input').value.toLowerCase().trim();
            if (!val) {
                viewNodes.refresh();
                return;
            }
            
            const matched = rawNodes.get({
                filter: function (item) {
                    return item.id.toLowerCase().includes(val) || 
                           item.type.toLowerCase().includes(val) ||
                           (item.properties.name && item.properties.name.toLowerCase().includes(val));
                }
            });
            
            if (matched.length > 0) {
                focusOnNode(matched[0].id);
            }
        }
        
        window.addEventListener('load', () => {
            initNetwork();
        });
        
    </script>
</body>
</html>
"""

def render_graph(json_path: Path, html_path: Path, open_browser: bool = False) -> None:
    json_path = Path(json_path)
    html_path = Path(html_path)

    if not json_path.exists():
        print(f"Error: JSON graph file not found: {json_path}")
        sys.exit(1)

    print(f"Loading JSON graph data from: {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Conforming to graphs list schema or single graph object
    if "graphs" in data and isinstance(data["graphs"], list) and len(data["graphs"]) > 0:
        graph = data["graphs"][0]
    else:
        graph = data

    nodes_count = len(graph.get("nodes", []))
    links_count = len(graph.get("links", []))

    print(f"Found graph with {nodes_count} nodes and {links_count} links.")

    # Generate HTML content by injecting JSON
    html_content = HTML_TEMPLATE
    html_content = html_content.replace("__FILE_NAME__", json_path.name)
    html_content = html_content.replace("__GRAPH_DATA_JSON__", json.dumps(graph))

    print(f"Writing interactive HTML dashboard to: {html_path}...")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("Success! HTML page written successfully.")
    
    if open_browser:
        print("Opening dashboard in default web browser...")
        webbrowser.open("file://" + str(html_path.resolve()))

def main():
    parser = argparse.ArgumentParser(description="Render interactive vis-network HTML from a JSON graph file.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to the input JSON graph file")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to the output HTML file")
    parser.add_argument("--open", action="store_true", help="Open the output HTML in the browser immediately")

    args = parser.parse_args()
    render_graph(Path(args.input), Path(args.output), args.open)

if __name__ == "__main__":
    main()
